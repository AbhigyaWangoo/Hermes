from src.responder import QueryHandler
from src.client.dataset_retriever.kaggle import KaggleDatasetClient

if __name__ == "__main__":
    # client = KaggleDatasetClient()

    # print(client.list_datasets(""))
    atlas_example=[
        (
            'cmotions/NLrestaurantreviews',
            {
                'freq': 9,
                'name': 'cmotions/NLrestaurantreviews',
                'links': 'https://huggingface.co/datasets/cmotions/NL_restaurant_reviews',
                'dataset_summary': "The dataset contains restaurant reviews collected in 2019 from Dutch restaurants. It is formatted using the DatasetDict format and includes train, test, and validation indices with 116693, 14587, and 14587 records respectively. The dataset includes both restaurant and review level information. \n\nRestaurant level information includes a unique restaurant ID, a Michelin star indicator, and scores for total, food, service, and decor. Review level information includes a unique review ID, a label for the reviewer's frequency of posting, scores for food, service, ambiance, waiting, value for money, and noise, the full review text, and the total length of the review. \n\nThe dataset can be used to model restaurant scores or Michelin star holders. For example, the review texts were used in a blog series to predict the next Michelin star restaurants using R."
            }
        ),
        (
            'prasadsawant7/sentimentanalysispreprocesseddataset',
            {
                'freq': 1,
                'name': 'prasadsawant7/sentimentanalysispreprocesseddataset',
                'links': 'https://huggingface.co/datasets/prasadsawant7/sentiment_analysis_preprocessed_dataset',
                'dataset_summary': "The dataset is designed for Text Classification, specifically Multi Class Classification, to train a model for Sentiment Analysis. It also allows for retraining the model based on feedback on incorrect sentiment predictions. The main features of the dataset are 'text' and 'labels'. 'Text' contains various forms of text like sentences, tweets, etc., while 'labels' contains numeric values representing sentiments - 0 for Negative, 1 for Neutral, and 2 for Positive. \n\nThe dataset also includes other features like 'preds', 'feedback', 'retrain_labels', and 'retrained_preds'. 'Preds' stores all predictions, 'feedback' allows users to indicate if a prediction is correct or not, 'retrain_labels' allows users to provide the correct label as feedback for retraining the model, and 'retrained_preds' stores all predictions after the feedback loop."
            }
        )
    ]
    query="restaurants with cuisine, price, and name"
    print(QueryHandler().coalesce_response(atlas_example, query))
