import easyocr
import os

def test_ocr():
    print("Initializing EasyOCR with custom model...")
    try:
        # Load the custom model
        # We specify recog_network='ko_webtoon_v2' which matches the filenames we created
        reader = easyocr.Reader(['ko'], recog_network='ko_webtoon_v2')
        print("Model loaded successfully!")
        
        # Test with a dummy image or just check initialization
        # If it initializes without error, the network structure matches the weights.
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ocr()
