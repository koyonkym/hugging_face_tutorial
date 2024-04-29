# Hugging Face Tutorials

This repository contains code examples to demonstrate various tutorials related to Hugging Face Transformers.

## Instructions for Training Models in Google Colab:

1. Clone the repository:
    ```bash
    !git clone https://github.com/koyonkym/hugging_face_tutorial.git
    ```

2. Install dependencies:
    ```bash
    !pip install -r hugging_face_tutorial/requirements.txt
    ```

3. Run the desired script. For example, to train a model:
    ```bash
    !python hugging_face_tutorial/transformers/training.py
    ```

4. Zip the output folder:
    ```bash
    !zip -r test_trainer.zip test_trainer
    ```

5. Download the zip file:
    ```python
    from google.colab import files
    files.download("test_trainer.zip")
    ```

Feel free to explore the code and experiment with the Hugging Face Transformers library!