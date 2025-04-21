import modelbit, sys
import random

# main function
def predict_weather(days_from_now: int):
  prediction = random.choice(["sunny", "cloudy", "just right"])
  return {
    "weather": prediction,
    "message": f"In {days_from_now} days it will be {prediction}!"
  }

# to run locally via git & terminal, uncomment the following lines
# if __name__ == "__main__":
#   result = predict_weather(...)
#   print(result)