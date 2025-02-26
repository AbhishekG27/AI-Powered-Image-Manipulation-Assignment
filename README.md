# 👟 Advanced Shoe Replacement & Editor

An AI-powered web application that allows users to replace shoes in images, change shoe colors, modify backgrounds, and apply photo effects with professional-quality results.


## 🌟 Features

- **Intelligent Shoe Detection**: Uses the Segment Anything Model (SAM) to automatically detect shoes in images with high precision
- **Realistic Shoe Replacement**: Seamlessly replaces shoes while preserving lighting, shadows, and perspective
- **Color Editing**: Change shoe colors while maintaining texture and material details
- **Background Manipulation**: Replace or modify backgrounds with solid colors or custom images
- **Photo Effects**: Apply professional visual effects like Vintage, Cool, Warm, High Contrast, and Black & White
- **User-Friendly Interface**: Simple, intuitive Streamlit-based UI for easy editing

## 📋 Requirements

- Python 3.7+
- OpenCV
- PyTorch
- Streamlit
- Segment Anything Model (SAM)
- NumPy
- Matplotlib
- SciPy
- PIL (Pillow)

## 🚀 Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/shoe-replacement-editor.git
   cd shoe-replacement-editor
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   streamlit run app.py
   ```

The application will automatically download the SAM model weights on first run if they are not already present.

## 📊 How It Works

### Shoe Replacement

The application uses a multi-step process to achieve realistic shoe replacements:

1. **Segmentation**: The Segment Anything Model (SAM) precisely identifies shoes in both the original and replacement images
2. **Alignment**: The replacement shoe is scaled, rotated, and positioned to match the original shoe's orientation
3. **Advanced Blending**: Sophisticated blending algorithms create seamless transitions between the replacement shoe and the background
4. **Lighting & Shadow Adjustment**: Environmental lighting is analyzed and replicated on the new shoe, including shadow effects
5. **Final Polish**: Subtle enhancements like color grading and detail preservation are applied for a professional finish

### Shoe Editing

The editor provides several powerful modification capabilities:

1. **Color Changing**: Modifies the shoe's color while preserving its texture, material properties, and highlights
2. **Background Replacement**: Precisely extracts the shoe and places it on a new background
3. **Effect Application**: Implements various visual styles through color space transformations and filter effects

## 🖼️ Example Usage

### Shoe Replacement

1. Upload an image containing shoes you want to replace
2. Upload an image of the replacement shoe
3. Click "Replace Shoe" and wait for processing to complete
4. Download the resulting image with professionally replaced shoes

### Shoe Editing

1. Upload an image containing shoes
2. Choose from color change, background replacement, or effects
3. Adjust settings as desired and apply changes
4. Download your edited image

## 🔧 Technical Details

- **Image Processing**: Uses OpenCV for most image manipulation operations
- **Deep Learning**: Leverages the Segment Anything Model (SAM) from Meta AI for precise segmentation
- **Color Science**: Implements advanced color space transformations (RGB, HSV, LAB) for realistic edits
- **Edge Processing**: Uses specialized algorithms for realistic edge transitions and blending

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgements

- [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) by Meta AI Research
- [Streamlit](https://streamlit.io/) for the interactive web interface
- [OpenCV](https://opencv.org/) for computer vision capabilities
- [PyTorch](https://pytorch.org/) for deep learning functionalities

## 📞 Contact

If you have any questions or feedback, please open an issue on GitHub or contact [aasthab@gmail.com][2022abhishek.g@vidyashilp.edu.in].

---

Made with ❤️ using Computer Vision and Deep Learning
