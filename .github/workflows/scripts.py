import os

# Access the GitHub secret from the environment
secret_value = os.getenv("OPENAI_API_KEY ")

if secret_value:
    print(f"Secret successfully retrieved:{secret_value }")
else:
    print("Secret not found. Ensure it's set in the GitHub Actions secrets.")