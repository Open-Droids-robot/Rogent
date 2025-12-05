import sys
print(f"Python Version: {sys.version}")

try:
    import google.generativeai as genai
    print("SUCCESS: google.generativeai imported.")
    print(f"Version: {genai.__version__}")
except ImportError as e:
    print(f"ERROR: Failed to import google.generativeai: {e}")
    sys.exit(1)

try:
    from google.generativeai import protos
    print("SUCCESS: google.generativeai.protos imported.")
except ImportError as e:
    print(f"ERROR: Failed to import google.generativeai.protos: {e}")
    # Check if it's available under a different path or if the package is corrupted
    try:
        import google.ai.generativelanguage as glm
        print("INFO: google.ai.generativelanguage is available (underlying protos).")
    except ImportError:
        print("ERROR: google.ai.generativelanguage is ALSO missing.")
    sys.exit(1)

print("\nEnvironment seems OK for Gemini SDK.")
