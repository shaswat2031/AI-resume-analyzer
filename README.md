# 🔍 AI Resume Analyzer

A powerful AI-powered resume analysis and tailoring tool built with Streamlit. This application helps job seekers optimize their resumes by analyzing them against job descriptions, providing detailed ATS (Applicant Tracking System) scores, and generating tailored resumes to improve job application success rates.

## ✨ Features

### 📊 Resume Analysis
- **ATS Score**: Get a comprehensive 0-100 score showing how well your resume matches the job description
- **Strengths & Weaknesses**: Detailed breakdown of your resume's strong points and areas for improvement
- **Missing Keywords**: Identify important keywords from the job description that are missing in your resume
- **Improvement Tips**: Actionable suggestions to enhance your resume's effectiveness

### 📈 Advanced Visualizations
- **Radar Chart**: Visual breakdown of ATS scores across different categories:
  - Content Quality (Achievement Focus, Action Verbs, Quantified Results)
  - Format & Structure (ATS Readability, Section Headers, Bullet Points)
  - Experience Match (Job Alignment, Skills Coverage, Required Qualifications)
- **Skills Match Analysis**: Visual representation of matched vs missing keywords

### ✍️ Resume Tailoring
- **AI-Powered Tailoring**: Generate a customized resume optimized for the specific job description
- **Multiple Download Formats**: Download tailored resumes in TXT and DOCX formats
- **Cover Note Generation**: Auto-generate personalized cover notes highlighting key achievements

### 🤖 Multiple AI Models
Choose from different AI models for analysis:
- **GPT-3.5**: Fast & efficient analysis
- **GPT-4o**: More detailed analysis
- **GPT-5**: Best results, limited usage
- **DeepSeek**: Specialized for resumes

## 🚀 Getting Started

### Prerequisites
- Python 3.7 or higher
- An OpenAI-compatible API key (supports ChatAnywhere API)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/shaswat2031/AI-resume-analyzer.git
   cd AI-resume-analyzer
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root and add your API key:
   ```env
   OPENAI_API_KEY=your_api_key_here
   ```
   
   Or set the environment variable directly:
   ```bash
   # Windows
   set OPENAI_API_KEY=your_api_key_here
   
   # macOS/Linux
   export OPENAI_API_KEY=your_api_key_here
   ```

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📋 Usage

1. **Upload Your Resume**: Upload your resume in PDF, DOCX, or TXT format
2. **Paste Job Description**: Copy and paste the complete job description you're targeting
3. **Choose AI Model**: Select your preferred AI model from the sidebar
4. **Analyze**: Click "🔎 Analyze & Score" to get detailed analysis and scoring
5. **Tailor**: Click "✍️ Generate Tailored Resume" to create an optimized version
6. **Download**: Save your tailored resume and cover note

## 📁 Project Structure

```
AI-resume-analyzer/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (API keys)
├── .gitignore         # Git ignore rules
└── README.md          # Project documentation
```

## 🔧 Configuration

### Supported File Formats
- **PDF**: Using `pdfplumber` for text extraction
- **DOCX**: Using `docx2txt` for Microsoft Word documents
- **TXT**: Plain text files

### API Configuration
The application uses OpenAI-compatible APIs with the following default settings:
- **API Base**: `https://api.chatanywhere.tech`
- **Default Model**: `gpt-3.5-turbo`
- **Temperature**: 0.0-0.3 (depending on the task)

## 🛠️ Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | 1.24.0 | Web application framework |
| openai | 0.27.8 | AI model integration |
| pdfplumber | 0.7.4 | PDF text extraction |
| python-docx | 0.8.11 | DOCX file creation |
| docx2txt | 0.8 | DOCX text extraction |
| plotly | 5.13.0 | Interactive visualizations |
| python-dotenv | 1.0.0 | Environment variable management |
| pillow | 10.4.0 | Image processing |

## 💡 Tips for Best Results

### Resume Upload
- Ensure your resume is well-formatted and ATS-friendly
- Use clear section headers (Experience, Education, Skills, etc.)
- Include quantified achievements where possible

### Job Description
- Paste the complete job description including:
  - Job responsibilities
  - Required skills and qualifications
  - Preferred experience
  - Company information

### Model Selection
- **GPT-3.5**: Best for quick analysis and general improvements
- **GPT-4o**: More detailed insights and better keyword detection
- **DeepSeek**: Specialized for resume-specific optimizations

## 🔒 Security & Privacy

- **API Keys**: Stored securely in environment variables
- **File Processing**: All file processing happens locally
- **Data Privacy**: No resume data is stored permanently
- **Temporary Files**: Automatically cleaned up after processing

## 🚧 Troubleshooting

### Common Issues

1. **API Key Error**
   - Ensure your API key is correctly set in the `.env` file
   - Verify the API key has sufficient credits/quota

2. **File Upload Issues**
   - Check file format (PDF, DOCX, TXT only)
   - Ensure file is not corrupted or password-protected

3. **Analysis Fails**
   - Try a shorter job description if the request is too large
   - Switch to a different AI model
   - Check your internet connection

### Error Messages
- **"API_KEY not set"**: Add your API key to the `.env` file
- **"Failed to extract text"**: Check if the file format is supported
- **"API call failed"**: Verify API key and internet connection

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for the web interface
- Powered by OpenAI-compatible APIs for AI analysis
- Visualizations created with [Plotly](https://plotly.com/)
- PDF processing with [pdfplumber](https://github.com/jsvine/pdfplumber)

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Open an issue on GitHub
3. Review existing issues for similar problems

---

**Made with ❤️ for job seekers everywhere**

*Help improve your job application success rate with AI-powered resume optimization!*