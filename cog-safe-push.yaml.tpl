model: ostris/flux-dev-lora-trainer
test_model: replicate-internal/test-flux-fine-tuner

train:
  train_timeout: 600
  test_cases:

    # tiny training with autocaption
    - inputs:
        input_images: https://replicate.delivery/pbxt/LbXqvBzG3Bfo6CDkIkBNo8a8v6GWZGQrqxBrgeRpaQ1knkyD/me-tiny.zip
        steps: 50
        autocaption: true
        trigger_word: NDRS
        hf_repo_id: replicate/test-flux-fine-tuner-integration
        hf_token: "$HF_TOKEN"
      match_prompt: "A dictionary of a version and weights which are a .tar file"
