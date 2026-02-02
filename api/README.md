# 🎨 AnimeGAN API

Transform your photos into stunning anime-style images using AnimeGANv2.

## 🚀 Features

- **Multiple Anime Styles**: Choose from Shinkai, Hayao, or Paprika models
- **Enhancement Options**: Basic, Enhanced, With Lines, or Maximum anime effect
- **RESTful API**: Easy to integrate with any application
- **Swagger Documentation**: Interactive API docs at `/docs`

## 📦 Installation

### 1. Clone the repository and navigate to the api folder

```bash
cd c:\Avatar\api
```

### 2. Create a virtual environment

```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up model checkpoints

Create a `checkpoints` folder and add the AnimeGANv2 model checkpoints:

```
api/
├── checkpoints/
│   ├── generator_Shinkai_weight/
│   │   ├── checkpoint
│   │   ├── generator.data-00000-of-00001
│   │   └── generator.index
│   ├── generator_Hayao_weight/
│   └── generator_Paprika_weight/
```

You can get the checkpoints from: https://github.com/TachibanaYoshino/AnimeGANv2

### 5. Run the API

```bash
python main.py
# or
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 📖 API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔗 API Endpoints

### `GET /`
Welcome endpoint with API information.

### `GET /health`
Health check endpoint.

### `GET /models`
List available anime style models and enhancement types.

### `POST /transform`
Transform a single image to anime style.

**Parameters:**
- `file`: Image file (JPEG/PNG)
- `model`: Anime model (`Shinkai`, `Hayao`, `Paprika`)
- `enhancement`: Enhancement type (`basic`, `enhanced`, `with_lines`, `maximum`)

**Example with cURL:**
```bash
curl -X POST "http://localhost:8000/transform?model=Shinkai&enhancement=enhanced" \
     -H "accept: image/jpeg" \
     -F "file=@your_photo.jpg" \
     --output anime_result.jpg
```

### `POST /transform/all`
Transform an image and get all 4 enhancement versions.

**Example with cURL:**
```bash
curl -X POST "http://localhost:8000/transform/all?model=Shinkai" \
     -F "file=@your_photo.jpg"
```

**Response:**
```json
{
    "request_id": "abc123...",
    "model": "Shinkai",
    "original_filename": "your_photo.jpg",
    "outputs": {
        "basic": "/outputs/abc123_basic.jpg",
        "enhanced": "/outputs/abc123_enhanced.jpg",
        "with_lines": "/outputs/abc123_with_lines.jpg",
        "maximum": "/outputs/abc123_maximum.jpg"
    },
    "message": "Successfully transformed image into all anime styles!"
}
```

### `GET /outputs/{filename}`
Retrieve a generated output image.

## 🎯 Example Usage with Python

```python
import requests

# Transform a single image
url = "http://localhost:8000/transform"
files = {"file": open("photo.jpg", "rb")}
params = {"model": "Shinkai", "enhancement": "enhanced"}

response = requests.post(url, files=files, params=params)

if response.status_code == 200:
    with open("anime_result.jpg", "wb") as f:
        f.write(response.content)
    print("Success! Check anime_result.jpg")
else:
    print(f"Error: {response.json()}")
```

## 🎨 Available Models

| Model | Style | Best For |
|-------|-------|----------|
| **Shinkai** | Makoto Shinkai style | Realistic anime, beautiful backgrounds |
| **Hayao** | Studio Ghibli style | Soft, painterly anime aesthetic |
| **Paprika** | Vibrant colors | High-contrast, vivid anime images |

## 🔧 Enhancement Types

| Type | Description |
|------|-------------|
| `basic` | Direct output from the model |
| `enhanced` | Enhanced colors, contrast, and sharpness |
| `with_lines` | Enhanced with anime-style line art |
| `maximum` | Maximum anime effect with all enhancements |

## 📁 Project Structure

```
api/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── checkpoints/        # Model checkpoint files
├── uploads/            # Temporary upload storage
└── outputs/            # Generated output images
```

## ⚠️ Notes

- The API uses TensorFlow 1.x compatibility mode
- Large images may take longer to process
- Ensure you have sufficient GPU memory for optimal performance

## 📄 License

This project is for educational purposes. AnimeGANv2 is licensed under MIT License.
