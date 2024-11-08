trainer = AnimeGeneratorTrainer("images")
trainer.load_checkpoint("checkpoints/checkpoint_epoch_X.pt")
generated_images = trainer.generate_samples(num_samples=4)