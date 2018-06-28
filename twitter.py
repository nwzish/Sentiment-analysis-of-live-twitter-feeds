from tweepy import Stream , OAuthHandler , API
from tweepy.streaming import StreamListener
import json
import connect
import classifier as c
import time



open("twitter-out.txt","w").close()
open("twitter-report.txt" , "w").close()
topic = connect.topic
class Listner(StreamListener):

    def on_data(self,data):
        try:
            all_data = json.loads(data)
            # print(all_data["text"])
            tweet = all_data["text"]
            print(tweet)
            sentiment , confidence = c.FindSentiment(tweet)
            print(sentiment,confidence,tweet)
            output = open("twitter-report.txt",'a')
            output.write(str(sentiment) +" , "+ str(tweet))
            output.write("\n")
            output.close()


            if confidence*100 >= 80:
                print("HELLO")
                sav = open("twitter-out.txt" , 'a')
                sav.write(sentiment)
                sav.write("\n")
                sav.close()

            return True
        except:

            return True

    def on_error(self, status):
        print(status)

def startSentiment():
    auth = OAuthHandler(connect.API_Key , connect.API_Secret)
    auth.set_access_token(connect.Access_Token , connect.Access_Token_Secret)

    twitterStream = Stream(auth,Listner())
    twitterStream.filter(track=[topic])
    time.sleep(1)

if __name__ == "__main__":
    startSentiment()



