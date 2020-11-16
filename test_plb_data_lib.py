import pytest
import pandas as pd
import plb_data_lib


def test_missing_normal(capfd):
    df = pd.DataFrame([[4, 2], [2, 5]], columns=["colonne_1", "colonne_2"])
    plb_data_lib.missing_data(df)
    out, err = capfd.readouterr()
    result = [
        "Le jeu de données contient 2 colonnes :",
        "- La colonne COLONNE_1 contient 0.0% de valeurs manquantes",
        "- La colonne COLONNE_2 contient 0.0% de valeurs manquantes",
        "Le jeu de données contient 2 observations",
        "Il y a 0 caractéristiques sur 2 qui ont plus de 10% de valeurs manquantes",
    ]
    for r, test in zip(out.split("\n"), result):
        assert r == test

def test_missing_limit(capfd):
    df = pd.DataFrame(columns=["colonne_1", "colonne_2"])
    plb_data_lib.missing_data(df)
    out, err = capfd.readouterr()
    assert out.strip("\n") == "Le jeu de données est vide"