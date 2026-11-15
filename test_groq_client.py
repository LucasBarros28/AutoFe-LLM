
from src.llm.groq_client import GroqClient


client = GroqClient()

print("Cliente Groq inicializado!")
print("Modelo:", client.model)

context = {
    "problem_type": "classification",

    "number_of_rows": 100,

    "columns": [
        "Age",
        "Fare",
        "FamilySize",
        "Embarked"
    ],

    "numerical_columns": [
        "Age",
        "Fare",
        "FamilySize"
    ],

    "column_information": {
        "Age": {
            "missing_values": 15,
            "min": 1,
            "skewness": 0.8
        },

        "Fare": {
            "missing_values": 0,
            "min": 0,
            "skewness": 4.5
        },

        "FamilySize": {
            "missing_values": 0,
            "min": 1,
            "skewness": 1.2
        },

        "Embarked": {
            "missing_values": 5
        }
    }
}

print("\nConsultando Groq...")

try:

    suggestions = client.suggest_features(context)

    print("\nSUGESTÕES RECEBIDAS:")

    for suggestion in suggestions:
        print(suggestion)

    print("\nSUGESTÕES REJEITADAS:")

    for rejected in client.last_rejected_suggestions_:
        print(rejected)

    print("\nTeste concluído!")

except Exception as error:

    print("\nERRO NO TESTE:")
    print(type(error).__name__, error)
