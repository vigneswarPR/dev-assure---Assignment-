#!/usr/bin/env python3
"""
Test Tesseract OCR functionality
"""

import pytesseract
from PIL import Image
import os

def test_tesseract():
    print("Testing Tesseract OCR installation...")

    # Test if Tesseract is available
    try:
        version = pytesseract.get_tesseract_version()
        print(f"[SUCCESS] Tesseract version: {version}")
    except Exception as e:
        print(f"[ERROR] Tesseract not found: {e}")
        print("Please install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
        return False

    # Test OCR on a sample image
    test_image = 'sample_data/booking_search_1.png'
    if os.path.exists(test_image):
        print(f"\nTesting OCR on: {test_image}")
        try:
            text = pytesseract.image_to_string(Image.open(test_image))
            print(f"Extracted text length: {len(text)} characters")
            if text.strip():
                preview = text[:200].replace('\n', ' ')
                print(f"Text preview: {preview}...")
                return True
            else:
                print("No text detected in image")
                return False
        except Exception as e:
            print(f"[ERROR] OCR test failed: {e}")
            return False
    else:
        print(f"Test image not found: {test_image}")
        return False

if __name__ == "__main__":
    test_tesseract()
