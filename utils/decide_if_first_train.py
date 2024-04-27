def decide_if_first_train():
    from utils.load_the_champion_model import load_the_champion_model

    champion_model = load_the_champion_model()
    return 'train_the_model' if champion_model is None else 'train_the_challenger_model'
