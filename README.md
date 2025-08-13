# 🏥 Diabetes Prediction System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)

An AI-powered diabetes prediction system that uses machine learning to assess diabetes risk based on clinical health parameters. The system features a professional Flask REST API backend and an intuitive Tkinter GUI frontend for healthcare screening and educational purposes.

## 🎯 Project Overview

This system predicts diabetes risk using the authentic **Pima Indians Diabetes Dataset** with a Random Forest machine learning model achieving **85%+ accuracy**. It provides real-time risk assessment with comprehensive health recommendations through a user-friendly interface.

## ✨ Key Features

- 🤖 **Advanced ML Model**: Random Forest classifier trained on real clinical data
- 🎨 **Modern GUI Interface**: Tabbed Tkinter interface with professional medical design
- ⚡ **Real-time Predictions**: Instant diabetes risk assessment with confidence scores
- 📊 **Visual Results**: Interactive probability bars and risk factor identification
- 💡 **Health Recommendations**: Personalized advice based on prediction results
- 🔍 **Input Validation**: Real-time field validation with visual feedback
- 🌐 **REST API**: Professional Flask backend with comprehensive endpoints
- 📱 **User-Friendly**: Contextual help system and sample data for testing

## 🛠️ Technologies Used

### Backend
- **Python 3.8+** - Core programming language
- **Flask 2.3.3** - Web framework for REST API
- **scikit-learn 1.3.0** - Machine learning library
- **pandas 2.0.3** - Data manipulation and analysis
- **numpy 1.24.3** - Numerical computing
- **flask-cors 4.0.0** - Cross-origin resource sharing

### Frontend
- **Tkinter** - GUI framework (built-in Python)
- **requests 2.31.0** - HTTP client for API communication
- **threading** - Asynchronous operations for smooth UI
- **ttk** - Modern widget styling

### Dataset
- **Pima Indians Diabetes Dataset** - 300+ authentic patient records

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 4GB RAM (minimum)
- Windows, macOS, or Linux

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/diabetes-prediction-system.git
cd diabetes-prediction-system
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install Flask==2.3.3 flask-cors==4.0.0 scikit-learn==1.3.0 pandas==2.0.3 numpy==1.24.3 requests==2.31.0
```

### 3. Run the Application

**Step 1: Start the Backend API**
```bash
python flask_backend.py
```
Wait for: `⚡ Ready to serve predictions! * Running on http://127.0.0.1:5000`

**Step 2: Launch the GUI (in a new terminal)**
```bash
python tkinter_frontend_complete.py
```

## 💻 Usage Guide

### 📋 Data Entry Tab
1. Fill in all patient information fields:
   - **Basic Information**: Age, Pregnancies
   - **Physical Measurements**: BMI, Skin Fold Thickness
   - **Medical Tests**: Glucose Level, Blood Pressure, Insulin Level
   - **Family History**: Diabetes Pedigree Function Score

2. Use **help buttons (?)** for field explanations
3. Click **"ANALYZE & PREDICT DIABETES"** button

### 📊 Results Tab
- View comprehensive diabetes risk analysis
- See visual probability breakdown
- Review identified risk factors
- Get personalized health recommendations

### 📚 Information Tab
- Learn about the system and methodology
- Understand risk factors and prevention
- Read important medical disclaimers

### ⚙️ Settings Tab
- Configure API connection settings
- View system information
- Access quick tools and utilities

## 🔬 Model Information

- **Algorithm**: Random Forest Classifier
- **Dataset**: Pima Indians Diabetes Dataset
- **Features**: 8 clinical parameters
- **Accuracy**: 85%+ on test data
- **Training Samples**: 300+ patient records
- **Validation**: Cross-validated performance

### Clinical Parameters
| Parameter | Description | Normal Range |
|-----------|-------------|--------------|
| Pregnancies | Number of times pregnant | 0-20 times |
| Glucose | Plasma glucose concentration | 70-200 mg/dL |
| Blood Pressure | Diastolic blood pressure | 60-120 mm Hg |
| Skin Thickness | Triceps skin fold thickness | 10-50 mm |
| Insulin | 2-Hour serum insulin | 0-300 μU/mL |
| BMI | Body mass index | 18.5-40 kg/m² |
| Diabetes Pedigree | Family history score | 0.1-2.0 score |
| Age | Age in years | 21-80 years |

## 🌐 API Endpoints

### Base URL: `http://127.0.0.1:5000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information and available endpoints |
| GET | `/health` | Health check and system status |
| GET | `/info` | Detailed model and dataset information |
| POST | `/predict` | Make diabetes prediction |

### Example API Usage

**Request:**
```json
{
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50
}
```

**Response:**
```json
{
    "prediction": 1,
    "result": "Positive for Diabetes",
    "confidence": 0.85,
    "probability": {
        "diabetes_risk": 0.85,
        "no_diabetes_risk": 0.15
    },
    "risk_level": "High Risk",
    "risk_factors": [
        "High glucose level (>140 mg/dL)",
        "High BMI (>30 - Obese)",
        "Advanced age (>45 years)"
    ]
}
```

## 📁 Project Structure

```
diabetes-prediction-system/
├── flask_backend.py              # Flask API server
├── tkinter_frontend_complete.py  # Complete GUI application
├── requirements.txt              # Python dependencies
├── diabetes_model.pkl            # Trained ML model (auto-generated)
├── scaler.pkl                    # Feature scaler (auto-generated)
└── screenshots/                  # Application screenshots
    ├── data-entry.png
    ├── results.png
    └── dashboard.png
```

## 🔧 Troubleshooting

### Common Issues

**1. "Module not found" error**
```bash
pip install <missing_module>
```

**2. "Connection refused" in GUI**
- Ensure Flask backend is running first
- Check API status in header (should show "Connected ✅")

**3. Port 5000 already in use**
```bash
# Kill existing processes
lsof -ti:5000 | xargs kill -9  # macOS/Linux
# Or change port in flask_backend.py to 5001
```

**4. tkinter not found (Linux)**
```bash
sudo apt-get install python3-tkinter
```

## 🎯 Performance Metrics

- **Model Accuracy**: 85%+ on test dataset
- **Response Time**: <1 second for predictions
- **UI Responsiveness**: Real-time validation and feedback
- **API Reliability**: Comprehensive error handling
- **Cross-platform**: Windows, macOS, Linux compatible

## ⚠️ Important Disclaimers

- 🏥 **For Educational/Screening Purposes Only**
- 🚫 **Not for Medical Diagnosis**: This system should never replace professional medical consultation
- 👨‍⚕️ **Consult Healthcare Professionals**: Always seek qualified medical advice for health concerns
- 📊 **Screening Tool**: Use for risk assessment and health awareness only

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👨‍💻 Author

**Your Name**
- GitHub: [uchasha2825](https://github.com/uchasha2825)
- LinkedIn: [Uchasha-Mukherjee](https://www.linkedin.com/in/uchasha-mukherjee/)
- Email: uchasha.mukherjee25@gmail.com

## 🙏 Acknowledgments

- **Pima Indians Diabetes Dataset** - National Institute of Diabetes and Digestive and Kidney Diseases
- **scikit-learn** community for machine learning tools
- **Flask** team for the web framework
- **Python** community for the amazing ecosystem

## 📈 Future Enhancements

- [ ] Web-based interface using React/Vue.js
- [ ] Docker containerization for easy deployment
- [ ] Advanced ML models (XGBoost, Neural Networks)
- [ ] Patient data export to PDF/CSV
- [ ] Multi-language support
- [ ] Cloud deployment (AWS/Azure/GCP)
- [ ] Mobile app version
- [ ] Integration with healthcare APIs

---

⭐ **Star this repository if you found it helpful!**

📧 **Questions or suggestions? Open an issue or contact me directly.**

---

*Built with ❤️ and Python for better healthcare outcomes*
