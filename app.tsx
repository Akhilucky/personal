import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Database, Image, Brain, FileJson, CheckCircle } from 'lucide-react';

const PipelineArchitecture = () => {
  const [expandedStages, setExpandedStages] = useState({});
  const [selectedComponent, setSelectedComponent] = useState(null);

  const toggleStage = (stageId) => {
    setExpandedStages(prev => ({
      ...prev,
      [stageId]: !prev[stageId]
    }));
  };

  const pipeline = [
    {
      id: 1,
      name: "Image Acquisition",
      icon: <Image className="w-6 h-6" />,
      color: "bg-blue-500",
      components: [
        {
          name: "Coordinate Buffer Manager",
          tech: "GeoPandas + Shapely",
          input: "CSV with (lat, lon)",
          output: "Buffered geometry (±20m polygon)",
          purpose: "Handle coordinate jitter by creating search area",
          code: `# Create 20m buffer around point
from shapely.geometry import Point
import geopandas as gpd

point = Point(lon, lat)
buffer = point.buffer(0.0002)  # ~20m
gdf = gpd.GeoDataFrame(geometry=[buffer])`
        },
        {
          name: "Building Footprint Fetcher",
          tech: "Overpass API / Google Buildings",
          input: "Buffered polygon",
          output: "Precise roof outline geometry",
          purpose: "Isolate actual roof region to reduce false positives",
          code: `# Query OSM for buildings in buffer
overpass_query = f"""
[out:json];
(
  way["building"]({lat-0.0002},{lon-0.0002},
                   {lat+0.0002},{lon+0.0002});
);
out geom;
"""`
        },
        {
          name: "Satellite Imagery API",
          tech: "Google Maps Static / Mapbox",
          input: "Roof outline + zoom level",
          output: "High-res RGB image (4096×4096)",
          purpose: "Fetch actual imagery for the roof region",
          code: `# Google Static Maps API
url = f"https://maps.googleapis.com/maps/api/staticmap?
center={lat},{lon}&zoom=20&size=640x640
&maptype=satellite&key={API_KEY}"`
        },
        {
          name: "Image Quality Checker",
          tech: "OpenCV + custom metrics",
          input: "Downloaded image",
          output: "QC flags (cloud %, resolution, date)",
          purpose: "Auto-flag NOT_VERIFIABLE cases early",
          code: `# Check image quality
cloud_pct = detect_cloud_cover(image)
resolution = estimate_gsd(image, zoom)
if cloud_pct > 20 or resolution > 0.3:
    return "NOT_VERIFIABLE"`
        }
      ],
      dataFlow: "CSV → Geometry Buffer → Building Footprints → Satellite Tile → Quality Check"
    },
    {
      id: 2,
      name: "Solar Panel Detection",
      icon: <Brain className="w-6 h-6" />,
      color: "bg-purple-500",
      components: [
        {
          name: "Pre-processing Pipeline",
          tech: "Albumentations + torchvision",
          input: "Raw satellite image",
          output: "Normalized tensor (C×H×W)",
          purpose: "Standardize imagery for model input",
          code: `import albumentations as A

transform = A.Compose([
    A.Resize(1024, 1024),
    A.Normalize(mean=[0.485, 0.456, 0.406]),
    A.ToTensorV2()
])`
        },
        {
          name: "Primary Detector (YOLOv11)",
          tech: "Ultralytics YOLOv11",
          input: "Preprocessed image",
          output: "Bounding boxes + class scores",
          purpose: "Fast initial detection of panel regions",
          code: `from ultralytics import YOLO

model = YOLO('yolov11_solar.pt')
results = model.predict(
    image, 
    conf=0.25,
    iou=0.45
)
boxes = results[0].boxes`
        },
        {
          name: "Segmentation Refiner (SAM 2)",
          tech: "Meta SAM 2",
          input: "YOLO bounding boxes",
          output: "Precise polygon masks",
          purpose: "Get exact panel boundaries for area calculation",
          code: `from sam2.sam2_image_predictor import SAM2ImagePredictor

predictor = SAM2ImagePredictor.from_pretrained("sam2_hiera_large")
masks = predictor.predict(
    image,
    point_coords=box_centers,
    point_labels=[1]*len(boxes)
)`
        },
        {
          name: "Confidence Calibrator",
          tech: "Temperature Scaling",
          input: "Raw model scores",
          output: "Calibrated probability [0-1]",
          purpose: "Produce reliable confidence scores for QC",
          code: `# Apply temperature scaling
import torch.nn.functional as F

calibrated = F.softmax(
    logits / temperature, 
    dim=-1
)
confidence = calibrated.max().item()`
        }
      ],
      dataFlow: "Image → Preprocess → YOLO Detection → SAM Segmentation → Calibrated Scores"
    },
    {
      id: 3,
      name: "RAG Knowledge Retrieval",
      icon: <Database className="w-6 h-6" />,
      color: "bg-green-500",
      components: [
        {
          name: "Multimodal Embedder",
          tech: "OpenCLIP / Jina v3",
          input: "Detected roof region + metadata",
          output: "768-dim embedding vector",
          purpose: "Convert visual+text into searchable vectors",
          code: `import open_clip

model, preprocess = open_clip.create_model_and_transforms('ViT-B-32')
image_emb = model.encode_image(preprocess(crop))
text_emb = model.encode_text(tokenize(metadata))`
        },
        {
          name: "Vector Database",
          tech: "Qdrant / Pinecone",
          input: "Query embedding",
          output: "Top-K similar verified cases",
          purpose: "Find historical examples to support QC decision",
          code: `from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)
hits = client.search(
    collection_name="verified_solar",
    query_vector=embedding,
    limit=5
)`
        },
        {
          name: "Context Aggregator",
          tech: "LangChain / LlamaIndex",
          input: "Retrieved similar cases",
          output: "Structured context document",
          purpose: "Prepare evidence for LLM reasoning",
          code: `context = {
    'similar_cases': [
        {
            'has_solar': hit.payload['has_solar'],
            'panel_count': hit.payload['count'],
            'notes': hit.payload['qc_notes']
        } for hit in hits
    ],
    'detection_features': features
}`
        }
      ],
      dataFlow: "Detection → Embed → Vector Search → Retrieve Similar → Aggregate Context"
    },
    {
      id: 4,
      name: "Explainability & QC",
      icon: <CheckCircle className="w-6 h-6" />,
      color: "bg-orange-500",
      components: [
        {
          name: "Vision-Language Model",
          tech: "GPT-4o / LLaVA 1.6",
          input: "Image + detection + RAG context",
          output: "Natural language reasoning",
          purpose: "Generate human-readable reason codes",
          code: `prompt = f"""
Analyze this rooftop image with {panel_count} detected panels.
Similar verified cases: {context['similar_cases']}

Provide 3 reason codes explaining the detection:
1. Visual evidence (module grid, shadows, reflections)
2. Comparison to similar cases
3. Confidence assessment
"""
response = llm.generate(prompt, image)`
        },
        {
          name: "QC Status Classifier",
          tech: "Rule Engine + ML",
          input: "Confidence, quality metrics, LLM output",
          output: "VERIFIABLE / NOT_VERIFIABLE",
          purpose: "Final quality control decision",
          code: `def classify_qc(confidence, quality, reasoning):
    if quality['cloud_pct'] > 20:
        return "NOT_VERIFIABLE"
    if confidence < 0.4:
        return "NOT_VERIFIABLE"
    if "unclear" in reasoning.lower():
        return "NOT_VERIFIABLE"
    return "VERIFIABLE"`
        },
        {
          name: "Reason Code Extractor",
          tech: "Structured Output Parsing",
          input: "LLM reasoning text",
          output: "List of codes ['module_grid', 'shadows']",
          purpose: "Create machine-readable audit trail",
          code: `import instructor
from pydantic import BaseModel

class QCOutput(BaseModel):
    reason_codes: list[str]
    qc_notes: list[str]
    
response = instructor.from_llm(llm).create(
    response_model=QCOutput,
    messages=[{"role": "user", "content": prompt}]
)`
        }
      ],
      dataFlow: "Detection + Context → VLM Reasoning → QC Classification → Extract Codes"
    },
    {
      id: 5,
      name: "Quantification & Output",
      icon: <FileJson className="w-6 h-6" />,
      color: "bg-red-500",
      components: [
        {
          name: "Panel Counter",
          tech: "Connected Components",
          input: "Segmentation masks",
          output: "Integer panel count",
          purpose: "Count distinct panels accurately",
          code: `from scipy.ndimage import label

labeled, num = label(binary_mask)
panel_count = num  # Number of connected regions`
        },
        {
          name: "Area Calculator",
          tech: "Polygon Area + GSD Scaling",
          input: "Polygon masks + image metadata",
          output: "PV area in m²",
          purpose: "Convert pixel area to real-world measurements",
          code: `from shapely.geometry import Polygon

polygon = Polygon(mask_contour)
pixel_area = polygon.area
gsd = 0.15  # Ground sampling distance (m/pixel)
area_sqm = pixel_area * (gsd ** 2)`
        },
        {
          name: "Capacity Estimator",
          tech: "Regression Model (XGBoost)",
          input: "Panel count, area, roof features",
          output: "Estimated kW",
          purpose: "Predict system capacity with uncertainty bounds",
          code: `import xgboost as xgb

features = [panel_count, area_sqm, roof_tilt, azimuth]
capacity_kw = model.predict([features])[0]

# Or simple heuristic:
capacity_kw = area_sqm * 0.20  # 200W/m²`
        },
        {
          name: "Audit Visualizer",
          tech: "OpenCV + PIL",
          input: "Image + masks + metadata",
          output: "Annotated PNG with overlays",
          purpose: "Create human-reviewable artifacts",
          code: `import cv2

overlay = image.copy()
cv2.polylines(overlay, [mask], True, (0,255,0), 2)
cv2.putText(overlay, f"Conf: {conf:.2f}", 
            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
cv2.imwrite('audit.png', overlay)`
        },
        {
          name: "JSON Formatter",
          tech: "Pydantic Models",
          input: "All computed attributes",
          output: "Structured JSON record",
          purpose: "Create API-ready, schema-validated output",
          code: `from pydantic import BaseModel
from datetime import datetime

class SolarDetectionOutput(BaseModel):
    sample_id: int
    lat: float
    lon: float
    has_solar: bool
    confidence: float
    panel_count_est: int
    pv_area_sqm_est: float
    capacity_kw_est: float
    qc_status: str
    qc_notes: list[str]
    bbox_or_mask: str
    timestamp: datetime
    
output = SolarDetectionOutput(**results)
with open('output.json', 'w') as f:
    f.write(output.model_dump_json(indent=2))`
        }
      ],
      dataFlow: "Masks → Count + Area → Capacity Estimate → Visualize → JSON Schema"
    }
  ];

  const ComponentDetail = ({ component }) => (
    <div className="bg-gray-800 rounded-lg p-4 mb-3 border border-gray-700">
      <div className="flex justify-between items-start mb-2">
        <div>
          <h4 className="font-semibold text-white">{component.name}</h4>
          <p className="text-sm text-blue-400">{component.tech}</p>
        </div>
      </div>
      
      <div className="grid grid-cols-2 gap-2 mb-3 text-sm">
        <div>
          <span className="text-gray-400">Input:</span>
          <p className="text-gray-200">{component.input}</p>
        </div>
        <div>
          <span className="text-gray-400">Output:</span>
          <p className="text-gray-200">{component.output}</p>
        </div>
      </div>
      
      <div className="mb-3">
        <span className="text-gray-400 text-sm">Purpose:</span>
        <p className="text-gray-200 text-sm">{component.purpose}</p>
      </div>
      
      <details className="mt-2">
        <summary className="text-sm text-green-400 cursor-pointer hover:text-green-300">
          View Code Example
        </summary>
        <pre className="bg-gray-900 p-3 rounded mt-2 text-xs text-green-300 overflow-x-auto">
          {component.code}
        </pre>
      </details>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
            Rooftop Solar Detection Pipeline
          </h1>
          <p className="text-gray-400">End-to-end architecture with 5 core stages</p>
        </div>

        {/* Data Flow Overview */}
        <div className="bg-gradient-to-r from-gray-800 to-gray-700 rounded-lg p-6 mb-8 border border-gray-600">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <svg className="w-6 h-6 text-yellow-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            Complete Data Flow
          </h2>
          <div className="bg-gray-900 rounded p-4 font-mono text-sm text-green-400">
            CSV Input → Buffer Geometry → Satellite Imagery → YOLOv11 Detection → SAM Segmentation → 
            Embedding → Vector Search → LLM Reasoning → QC Classification → Area/Capacity Calculation → 
            JSON Output + Audit PNG
          </div>
        </div>

        {/* Pipeline Stages */}
        <div className="space-y-6">
          {pipeline.map((stage) => (
            <div key={stage.id} className="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
              {/* Stage Header */}
              <div 
                className={`${stage.color} p-4 cursor-pointer hover:opacity-90 transition-opacity`}
                onClick={() => toggleStage(stage.id)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="bg-white bg-opacity-20 p-2 rounded">
                      {stage.icon}
                    </div>
                    <div>
                      <h3 className="text-xl font-bold">Stage {stage.id}: {stage.name}</h3>
                      <p className="text-sm opacity-90">{stage.components.length} components</p>
                    </div>
                  </div>
                  {expandedStages[stage.id] ? <ChevronDown /> : <ChevronRight />}
                </div>
              </div>

              {/* Stage Content */}
              {expandedStages[stage.id] && (
                <div className="p-6">
                  {/* Data Flow for this stage */}
                  <div className="mb-6 p-4 bg-gray-900 rounded border border-gray-700">
                    <h4 className="text-sm font-semibold text-gray-400 mb-2">Stage Data Flow:</h4>
                    <p className="text-sm text-blue-300 font-mono">{stage.dataFlow}</p>
                  </div>

                  {/* Components */}
                  <div className="space-y-4">
                    {stage.components.map((component, idx) => (
                      <ComponentDetail key={idx} component={component} />
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Integration Notes */}
        <div className="mt-8 bg-gradient-to-br from-purple-900 to-blue-900 rounded-lg p-6 border border-purple-500">
          <h2 className="text-2xl font-bold mb-4">🔗 How Stages Connect</h2>
          <div className="space-y-3 text-sm">
            <div className="flex gap-3">
              <span className="text-blue-400 font-bold">1→2:</span>
              <p>Image Acquisition downloads satellite tiles and crops them to the detected building footprint, feeding clean roof images to the detector.</p>
            </div>
            <div className="flex gap-3">
              <span className="text-purple-400 font-bold">2→3:</span>
              <p>Detection results (bounding boxes + masks) are embedded using CLIP/Jina, then sent to the vector database to find similar verified cases.</p>
            </div>
            <div className="flex gap-3">
              <span className="text-green-400 font-bold">3→4:</span>
              <p>Retrieved similar cases become context for the VLM, which generates explainable reason codes and assigns a QC status.</p>
            </div>
            <div className="flex gap-3">
              <span className="text-orange-400 font-bold">4→5:</span>
              <p>QC-approved detections flow to quantification: polygon masks are measured, capacity estimated, and results formatted as JSON.</p>
            </div>
            <div className="flex gap-3">
              <span className="text-red-400 font-bold">2+5:</span>
              <p>Original image + segmentation masks are combined in the visualizer to create audit-ready PNGs with overlays and metadata.</p>
            </div>
          </div>
        </div>

        {/* Technology Stack Summary */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <h3 className="font-bold mb-2 text-blue-400">Core ML Models</h3>
            <ul className="text-sm space-y-1 text-gray-300">
              <li>• YOLOv11 (detection)</li>
              <li>• SAM 2 (segmentation)</li>
              <li>• OpenCLIP (embeddings)</li>
              <li>• GPT-4o / LLaVA (reasoning)</li>
            </ul>
          </div>
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <h3 className="font-bold mb-2 text-green-400">Data Infrastructure</h3>
            <ul className="text-sm space-y-1 text-gray-300">
              <li>• Qdrant (vector DB)</li>
              <li>• PostgreSQL + PostGIS</li>
              <li>• Redis (task queue)</li>
              <li>• S3 / MinIO (storage)</li>
            </ul>
          </div>
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <h3 className="font-bold mb-2 text-purple-400">APIs & Services</h3>
            <ul className="text-sm space-y-1 text-gray-300">
              <li>• Google Maps Static</li>
              <li>• Overpass API (OSM)</li>
              <li>• FastAPI (backend)</li>
              <li>• Celery (async tasks)</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PipelineArchitecture;
