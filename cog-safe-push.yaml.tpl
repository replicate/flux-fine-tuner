model: ostris/flux-dev-lora-trainer
test_model: replicate-internal/test-flux-fine-tuner
predict:
  compare_outputs: false
  predict_timeout: 300
  test_cases:

    # basic
    - inputs:
        prompt: PLSM style, an image of a tiger, made of yellow, red, and purple goo plasma
        num_outputs: 1
        lora_scale: 1.1
        num_inference_steps: 28
        guidance_scale: 1.5
        seed: 5259
        output_format: jpg
        replicate_weights: https://replicate.delivery/yhqm/MHJmIF7zlLKXPNN9LKN5yxYUuC7SyzKjMBtqeVwUGdPCpqqJA/trained_model.tar
      match_url: https://storage.googleapis.com/replicate-test-flux-finer-tuner/test-outputs/base.jpg

    # img2img and custom size
    - inputs:
        prompt: "PLSM style, an image of a formula one car, made of yellow, red, and purple goo plasma"
        num_outputs: 1
        lora_scale: 1.0
        aspect_ratio: "custom"
        width: 500
        height: 256
        num_inference_steps: 20
        model: "dev"
        guidance_scale: 1.5
        replicate_weights: "https://replicate.delivery/yhqm/MHJmIF7zlLKXPNN9LKN5yxYUuC7SyzKjMBtqeVwUGdPCpqqJA/trained_model.tar"
        output_format: "jpg"
        output_quality: 90
        image: "https://storage.googleapis.com/cog-safe-push-public/fast-car.jpg"
        prompt_strength: 0.9
        seed: 12919
      match_url: https://replicate.delivery/yhqm/zekZzRptTet51UvOUHUH0fnT04taL9seSvRQuP0p7fPnvoIcC/out-0.jpg

    # inpainting
    - inputs:
        prompt: "PLSM style, an image of a tiger, made of yellow, red, and purple goo plasma"
        num_outputs: 1
        lora_scale: 0.8
        aspect_ratio: custom
        width: 500
        height: 256
        num_inference_steps: 20
        model: "dev"
        guidance_scale: 1.5
        replicate_weights: "https://replicate.delivery/yhqm/MHJmIF7zlLKXPNN9LKN5yxYUuC7SyzKjMBtqeVwUGdPCpqqJA/trained_model.tar"
        output_format: "jpg"
        output_quality: 90
        prompt_strength: 1.0
        image: "https://replicate.delivery/pbxt/LbXcDNOFtOMiBChe9QHHbeNJ5KcoTNfWEGmIxswhaexvXWEs/Screenshot%202024-09-11%20at%2014.36.07.png"
        mask: "https://replicate.delivery/pbxt/LbXcE8bXqlTqPWIDb3SFFuJzIUMKSUscnqF97XVv0K2coEwr/Screenshot%202024-09-11%20at%2014.37.41.png"
        seed: 64090
      match_url: https://storage.googleapis.com/replicate-test-flux-finer-tuner/test-outputs/inpaint.jpg

    # aspect ratio
    - inputs:
        prompt: PLSM style, an image of a tiger, made of yellow, red, and purple goo plasma
        aspect_ratio: "16:9"
        num_outputs: 1
        lora_scale: 1.1
        num_inference_steps: 28
        guidance_scale: 1.5
        seed: 16726
        output_format: jpg
        replicate_weights: https://replicate.delivery/yhqm/MHJmIF7zlLKXPNN9LKN5yxYUuC7SyzKjMBtqeVwUGdPCpqqJA/trained_model.tar
      match_url: https://storage.googleapis.com/replicate-test-flux-finer-tuner/test-outputs/aspect-ratio.jpg

    # schnell
    - inputs:
        prompt: PLSM style, an image of a tiger, made of yellow, red, and purple goo plasma
        num_outputs: 1
        lora_scale: 1.1
        num_inference_steps: 4
        model: schnell
        guidance_scale: 1.5
        seed: 16726
        output_format: jpg
        replicate_weights: https://replicate.delivery/yhqm/MHJmIF7zlLKXPNN9LKN5yxYUuC7SyzKjMBtqeVwUGdPCpqqJA/trained_model.tar
      match_url: https://replicate.delivery/yhqm/rQhMIg6dhLYVJ5gXWfZI4Ik1DF9cGV7pm8UN0DKmpWDfJGhTA/out-0.jpg

    # extra lora
    - inputs:
        prompt: PLSM style, an image of a man ANDRS, made of yellow, red, and purple goo plasma
        num_outputs: 1
        lora_scale: 0.8
        num_inference_steps: 20
        model: dev
        guidance_scale: 1.5
        seed: 413
        extra_lora: andreasjansson/flux-me
        extra_lora_scale: 1.1
        replicate_weights: https://replicate.delivery/yhqm/MHJmIF7zlLKXPNN9LKN5yxYUuC7SyzKjMBtqeVwUGdPCpqqJA/trained_model.tar
        output_format: jpg
        output_quality: 90
      match_url: https://storage.googleapis.com/replicate-test-flux-finer-tuner/test-outputs/extra-lora.jpg

  fuzz:
    fixed_inputs:
      replicate_weights: https://replicate.delivery/yhqm/MHJmIF7zlLKXPNN9LKN5yxYUuC7SyzKjMBtqeVwUGdPCpqqJA/trained_model.tar
    disabled_inputs:
      - extra_lora
    duration: 120
    iterations: 5

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
