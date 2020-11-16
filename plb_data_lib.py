# Exploratory data analysis

def missing_data(df, percent=10):
    """
    Show statistics and charts about missing data
    More data on features with more than {percent} missing values
    """
    if len(df.index)>0:
        print(f"Le jeu de données contient {len(df.columns)} colonnes :")
        values = []
        for col in df.columns:
            values.append([round((1-df[col].notna().sum()/len(df))*100, 2), col])
            print(f"- La colonne {col.upper()} contient {values[-1][0]}% de valeurs manquantes")
        print(f"Le jeu de données contient {len(df)} observations")
        values.sort(key=lambda x: x[0], reverse=True)
        values = [v for v in values if v[0]>percent]
        print(f"Il y a {len(values)} caractéristiques sur {len(df.columns)} qui ont plus de {percent}% de valeurs manquantes")
    else:
        print("Le jeu de données est vide")

def drop_empty(df, percent=10):
    """
    Remove columns with more than {percent} missing values
    """
    print(f"Suppresion des colonnes avec plus de {percent}% de données manquantes")
    new_df = df.copy()
    for col in df.columns:
        if round((1-df[col].notna().sum()/len(df))*100, 2)>percent:
            new_df.drop(col, inplace=True, axis=1)
    return new_df