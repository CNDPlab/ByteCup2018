from Predictor.Base import BaseDataPipe


class SouGouTokenizer(object):
    def __init__(self):
        pass

    def tokenize(self, line_str):
        line_token = [i for i in line_str]
        return line_token


class SouGouDataPipe(BaseDataPipe):
    """
    --predict_pipe(self, list_of_lines_str)
    --line_to_middle(self, line_str, is_train)
    --line_to_processed(self, line_token)
    --train_w2v(self, sentance_list, embedding_dim, min_count, num_works)
    """
    def __init__(self, tokenizer=SouGouTokenizer()):
        super(SouGouDataPipe, self).__init__(tokenizer)
        self.tokenizer = tokenizer

    def _clean_line(self, word_str):
        is_ustr(word_str.replace('<Paragraph>', ''))
        """
        word_str = word_str.lower()
        return word_str
        """
        word_str = word_str.lower()
        return word_str

    def _tokenize(self, line_str):
        """
        line_token = self.tokenizer.tokenize(line_str)
        :param line_str:
        :return:  line_token
        """
        line_token = self.tokenizer.tokenize(line_str)
        return line_token

    def _clean_line_token_post(self, line_token):
        return line_token


"""
    data = data[1]
    #data['article'] = is_ustr(data.article.replace('<Paragraph>', ''))
    data['article'] = is_ustr(remove(strq2b(data.article)))
    data['summarization'] = is_ustr(remove(data.summarization))
    data['article'] = data['article'][:400]
    if len(data['article']) > 350:
        data['article'] = data['article'][:data['article'].rfind('。')+1]
    data['article_char'] = ['<BOS>'] + [i for i in data['article']] + ['<EOS>']
    data['summarization_char'] = ['<BOS>'] + [i for i in data['summarization']] + ['<EOS>']
    del data['article'], data['summarization']
    line = {i: data[i] for i in data.keys()}
    return line
"""


def is_ustr(in_str):
    out_str = ''
    for i in range(len(in_str)):
        if is_uchar(in_str[i]):
            out_str = out_str+in_str[i]
        else:
            pass
    return out_str


def is_uchar(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
            return True
    """判断一个unicode是否是数字"""
    if uchar >= u'\u0030' and uchar<=u'\u0039':
            return True
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'\u0041' and uchar<=u'\u005a') or (uchar >= u'\u0061' and uchar<=u'\u007a'):
            return True
    if uchar in ('-', ',', '，', '。', '.', '?', ':', ';'):
            return True
    return False

def strq2b(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:#全角空格直接转换
            inside_code = 32
        if inside_code == 58380:
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):#全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring

def remove(text):
    text = text.replace('<Paragraph>', '。')
    text = text.replace('！', '。')
    text = text.replace('：', ':')
    #text = re.sub(r'\([^)]*\)', '', text)
    #text = re.sub(r'\{.*\}', '', text)
    #text = re.sub(r'\（.*\）', '', text)
    text = re.sub(r'\([^()]*\)', '', text)

    text = re.sub(r'\（[^（）]*\）', '', text)

    text = re.sub(r'\[[^]]*\]', '', text)

    text = re.sub(r'\{[^{}]*\}', '', text)
    text = re.sub(r'\{[^{}]*\}', '', text)

    text = re.sub(r'\【[^【】]*\】', '', text)

    return text