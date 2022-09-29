# Third Party Libraries
import ktrain

predictor = ktrain.load_predictor("./models/deberta_ktrain")
results = predictor.predict(
    [
        (
            "I've been searching for the answer for this for some time, but I still can't find any answer... Can anyone please explain to me what this is?",
            "Religion must have the answer",
        ),
        (
            "WSJ begins the Jeb Bush campaign for 2016",
            "time to get that shack in montana.",
        ),
    ]
)  # type: ignore


print(results)
