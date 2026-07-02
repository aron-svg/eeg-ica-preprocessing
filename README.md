# Python Development Template

A comprehensive Python template for beginning new development projects with pre-configured tools and environment setup.

## 🚀 Quick Start

This template provides everything you need to start a new Python project quickly and efficiently.

### Prerequisites
- Python 3.x installed on your system
- Git and Git LFS installed

### Setup Environment

Choose the appropriate initialization script for your operating system:

#### Windows, Linux/macOS
```bash
uv sync
```

These scripts will:
- Create a Python virtual environment (`venv`)
- Install all required packages from `pyproject.toml`
- Set up the development environment

## 📁 Project Structure

```
Template_OrigenesRD/
├── src/                    # Source code directory
│   ├── main.py            # Main application entry point
│   ├── logger_config.yaml # Logging configuration
│   ├── logger_init.py     # Logger initialization
│   └── logging_config.py  # Logging setup
├── data/                  # Image/Video/CSV assets (tracked by Git LFS)
├── .vscode/               # VS Code configuration
│   └── launch.json        # Debug configuration
├── requirements.txt       # Python dependencies
├── pyproject.toml         # Project configuration
├── .gitignore             # Git ignore rules
├── .gitattributes         # Git LFS configuration
└── .flake8                # Code linting configuration
```

## 📦 Included Dependencies

The template comes with these pre-configured packages:

- **requests** (~2.28.1) - HTTP library
- **pandas** (2.1.1) - Data manipulation and analysis
- **python-dotenv** (0.20.0) - Environment variable management
- **numpy** (1.26.0) - Numerical computing
- **git-lfs** (3.6.1) - Git Large File Storage
- **flake8** - Code linting
- **black** - Code formatting
- **pyyaml** - YAML parsing

## 🛠️ Development Tools

### Code Quality
- **Flake8**: Configured for code linting with custom rules
- **Black**: Code formatting (configured in pyproject.toml)
- **VS Code**: Debug configuration included

### Logging
- Pre-configured logging system with YAML configuration
- Structured logging setup for development and production

## 🗂️ Git LFS Configuration

This repository uses Git LFS (Large File Storage) for managing large files efficiently.

### What is Git LFS?
Git LFS replaces large files with text pointers inside Git, while storing the actual file contents on a remote server like GitHub LFS.

### Configured File Types
The following file types are automatically tracked by Git LFS:

#### Archives
- `.zip`, `.tar.gz`, `.7z`, `.rar`

#### Media Files
- **Video**: `.mp4`, `.avi`, `.mov`, `.mkv`
- **Audio**: `.mp3`, `.wav`, `.flac`
- **Images**: `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.tiff`, `.svg`, `.ico`

#### Data Files
- `.csv`, `.json`, `.xml`, `.sql`
- **Databases**: `.db`, `.sqlite`, `.sqlite3`

#### Machine Learning
- **Models**: `.pkl`, `.pickle`, `.h5`, `.hdf5`, `.pt`, `.pth`, `.onnx`, `.pb`, `.tflite`, `.joblib`
- **Data**: `.npy`, `.npz`

#### Documents
- `.pdf`, `.doc`, `.docx`, `.ppt`, `.pptx`, `.xls`, `.xlsx`

#### Binaries
- `.exe`, `.dll`, `.so`, `.dylib`

### Git LFS Commands

```bash
# Check LFS status
git lfs status

# List tracked files
git lfs ls-files

# Track additional file types
git lfs track "*.extension"

# Migrate existing files to LFS
git lfs migrate import --include="*.extension"

# Pull LFS files
git lfs pull

# Push LFS files
git lfs push origin main
```

## 🔧 Usage

1. **Clone this template**:
   ```bash
   git clone <repository-url>
   cd Template_OrigenesRD
   code . # TO OPEN VSCODE
   ```

2. **Initialize the environment**:
   - Windows: `init.bat`
   - Linux/macOS: `./init.sh`

3. **Activate the virtual environment**:
   - Windows: `venv\Scripts\activate`
   - Linux/macOS: `source venv/bin/activate`

4. **Start developing**:
   ```bash
   python src/main.py
   ```

## 🖥️ VS Code Integration

The template includes VS Code configuration for:
- Python debugging (launch.json)
- Code formatting and linting settings
- Integrated terminal support

## 📝 Git LFS Notes

1. **First time setup**: Git LFS is already initialized in this repository
2. **Automatic tracking**: New files matching the configured patterns will automatically use LFS
3. **Collaboration**: Team members need to have Git LFS installed and run `git lfs install` in their local repos

## 🔍 Troubleshooting

### Environment Issues
- Ensure Python 3.x is installed
- Check that pip is up to date
- Verify virtual environment activation

### Git LFS Issues
If you encounter issues:
1. Ensure Git LFS is installed: `git lfs version`
2. Verify LFS is initialized: `git lfs install`
3. Check tracking patterns: `git lfs track`
4. Verify file status: `git lfs status`

## 🐳 Docker Support

Simple Docker setup for easy containerized development and deployment.

### Quick Start

```bash
# Build and run with Docker Compose
docker-compose up

# Or build manually
docker build -t template-origenesrd .
docker run -p 8050:8050 template-origenesrd
```

### What's Included

- **Dockerfile**: Simple single-stage build
- **docker-compose.yml**: Basic orchestration
- **.dockerignore**: Essential exclusions

The container runs your app on port 8050 with live code reloading enabled for development.

## 🤝 Contributing

When contributing to projects based on this template:
1. Follow the established code style (enforced by flake8 and black)
2. Update pyproject.toml for new dependencies
3. Use meaningful commit messages
4. Large files will automatically be handled by Git LFS
5. Test Docker builds before submitting PRs

---

**Ready to develop!** This template provides a solid foundation for Python projects with modern development practices, tools, and containerization pre-configured.

