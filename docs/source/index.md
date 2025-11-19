
```{include} ../../README.md
:relative-images:
```
----

# Documentation

Welcome to Coqui TTS documentation! This comprehensive guide will help you get started with text-to-speech synthesis, train your own models, and contribute to the project.

## 📚 Documentation Structure

- **User Guides** - Get started and use TTS models
- **Developer Guides** - Contribute and develop
- **API Reference** - Technical documentation
- **Architecture** - Deep dive into the codebase

For a complete overview, see the [Documentation Index](../DOCUMENTATION_INDEX.md).

----

# Documentation Content
```{eval-rst}
.. toctree::
    :maxdepth: 2
    :caption: Get Started

    tutorial_for_nervous_beginners
    installation
    faq
    contributing

.. toctree::
    :maxdepth: 2
    :caption: Using 🐸TTS

    inference
    docker_images
    training_a_model
    finetuning
    configuration
    formatting_your_dataset
    what_makes_a_good_dataset
    tts_datasets

.. toctree::
    :maxdepth: 2
    :caption: Advanced Topics

    implementing_a_new_model
    implementing_a_new_language_frontend
    marytts

.. toctree::
    :maxdepth: 2
    :caption: Main Classes

    main_classes/trainer_api
    main_classes/audio_processor
    main_classes/model_api
    main_classes/dataset
    main_classes/gan
    main_classes/speaker_manager

.. toctree::
    :maxdepth: 2
    :caption: `tts` Models

    models/glow_tts.md
    models/vits.md
    models/forward_tts.md
    models/tacotron1-2.md
    models/overflow.md
    models/tortoise.md
    models/bark.md
    models/xtts.md

.. toctree::
    :maxdepth: 2
    :caption: `vocoder` Models

```
