# Problem background

> **Key takeaways.** PlantGuard AI is an academic image-classification prototype for visible plant conditions; it is not expert diagnosis.

The project explores whether a leaf photograph can support a preliminary classification of plant and visible condition. Real-world appearance varies with lighting, background, and weather, while different conditions can look similar. The intended workflow is image upload, model inference, confidence display, and an optional visual explanation. The system must describe only classes represented by the verified dataset and must not imply that a Grad-CAM map precisely segments disease.

The team documentation describes transfer learning and a FastAPI application as project-wide goals. Maureen's completed work covers dataset preparation, a scratch CNN baseline, augmentation comparison, and a Streamlit frontend prepared for the API contract.
