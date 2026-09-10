## Running the Project

### Local Setup (Windows / PyCharm)

1. **Install dependencies:**
```powershell
   pip install -r requirements.txt
```

2. **Prepare data splits:**
```powershell
   python scripts/prepare_data.py
```

3. **Train a model:**
```powershell
   python scripts/train_model.py --epochs 20 --batch_size 32 --lr 1e-4 --model_name efficientnet_b0
```

4. **Evaluate on test set:**
```powershell
   python -m src.training.evaluate --model_name efficientnet_b0
```

5. **Export to ONNX:**
```powershell
   python scripts/export_onnx.py --model_name efficientnet_b0
```

6. **Run inference on a single image:**
```powershell
   python scripts/run_inference.py "path/to/image.jpg" --top_k 3
```

7. **Run the API locally:**
```powershell
   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
   Visit `http://localhost:8000/docs` for interactive Swagger UI.

8. **Run tests:**
```powershell
   pytest tests/ -v
```

---

### Running with Docker

Requires: trained model already exported to `models_store/onnx/efficientnet_b0.onnx`,
and `data/splits/test.csv` present (used for class name mapping).

1. **Build the image:**
```bash
   docker build -t agroml-api:latest .
```

2. **Run the container:**
```bash
   docker run -d -p 8000:8000 --name agroml-api agroml-api:latest
```

3. **Verify it's running:**
```bash
   curl http://localhost:8000/health
```

4. **Send a prediction request:**
```bash
   curl -X POST "http://localhost:8000/predict" \
     -F "file=@path/to/image.jpg;type=image/jpeg"
```

5. **Stop and remove the container:**
```bash
   docker stop agroml-api && docker rm agroml-api
```

**Note:** The Docker image uses `requirements.api.txt` — a minimal dependency
set for inference only (FastAPI, onnxruntime, no PyTorch). Training must be
done outside the container using `requirements.txt`.