def train_model(train_X, train_y, model):
    """retourne le modèle entraîné"""
    model.fit(train_X,train_y)
    return model

