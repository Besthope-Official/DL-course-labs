import math
from itertools import chain
import pandas as pd
import torch
from train import return_classifier
from dataset import make_tensors1
from config import BATCH_SIZE


def get_test_set():
    test_set = pd.read_csv(r'data/test.tsv', sep='\t')
    PhraseId = test_set['PhraseId']
    test_Phrase = test_set['Phrase']
    return PhraseId, test_Phrase


def testModel():
    PhraseId, test_Phrase = get_test_set()
    sentiment_list = []  # 定义预测结果列表
    batchNum = math.ceil(PhraseId.shape[0] / BATCH_SIZE)
    classifier = return_classifier()
    with torch.no_grad():
        for i in range(batchNum):
            if i == batchNum - 1:
                # 处理最后不足BATCH_SIZE的情况
                phraseBatch = test_Phrase[BATCH_SIZE * i:]
            else:
                phraseBatch = test_Phrase[BATCH_SIZE * i:BATCH_SIZE * (i + 1)]
            inputs, seq_lengths, org_idx = make_tensors1(phraseBatch)
            output = classifier(inputs, seq_lengths)
            sentiment = output.max(dim=1, keepdim=True)[1]
            sentiment = sentiment[org_idx].squeeze(1)
            sentiment_list.append(sentiment.cpu().numpy().tolist())

    sentiment_list = list(chain.from_iterable(
        sentiment_list))  # 将sentiment_list按行拼成一维列表
    result = pd.DataFrame({'PhraseId': PhraseId, 'Sentiment': sentiment_list})
    result.to_csv('SA_predict.csv', index=False)


if __name__ == '__main__':
    testModel()
