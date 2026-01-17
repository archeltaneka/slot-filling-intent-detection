import logging
from src.utils import save_model

def run_training_loop(model, trainer, train_loader, val_loader, num_epochs, model_name, save_dir):
    logging.info(f"Starting training for {model_name}...")
    results = {}
    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch(train_loader)
        results = trainer.evaluate(val_loader)
        logging.info(
            f"Epoch {epoch+1}/{num_epochs}: Loss {train_loss:.4f} | "
            f"Intent Acc: {results['intent_accuracy']:.4f} | Slot F1: {results['slot_f1']:.4f}"
        )
    # save_model(model, save_dir, f'{model_name}.pth')
    logging.info(f"{model_name} saved to {save_dir}")
    return results