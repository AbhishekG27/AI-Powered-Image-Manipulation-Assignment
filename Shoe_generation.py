from huggingface_hub import InferenceClient
import matplotlib.pyplot as plt

# Initialize Hugging Face Inference Client
client = InferenceClient(
    provider="hf-inference",
    api_key="api"
)

# Generate Image
image = client.text_to_image(
    "Astronaut riding a horse",
    model="stabilityai/stable-diffusion-xl-base-1.0"
)

# Display the image
plt.imshow(image)
plt.axis("off")  # Hide axes
plt.show()

# Save the image
image.save("generated_image.jpg")
print("Image saved as 'generated_image.jpg'")
