import argparse
from models.bigram.model_train import BigramModelTrain, BigramModelGenerate

MODELS = {
    'bigram': (BigramModelTrain, BigramModelGenerate)
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=MODELS.keys(),
                        help='select which model architecture to train')

    args = parser.parse_args()

    ## training
    model_trainer, model_generator = MODELS[args.model]
    model, decoder = model_trainer()
    print(f"Trained {args.model.upper()} model with {sum(p.numel() for p in model.parameters())} parameters.")

    ## inference
    print(decoder(model_generator(model, next(model.parameters()).device, 500)))