import torch
from transformers import*
from bertviz import *
import warnings

class bertvizJupyter:
    """
    This is class customized for using bertviz library for
    GeneLLM and other derivatives for bert.
    Useage described in readme.md file.

    member functions:
    loadSentenceandBertprocess : used to load sentences and get 
                                 attentions from fine tuned bert
    headView : This is a function used to view the specific heads' 
               attentions in fine tuned bert.
    modelView: This function is used to view all the attentions of 
               all layers.
    neuronView : This function is used to view specific layer and 
                 specific head neuron 
    
    """
    
     def __init__(self,modelVersion,bertModel):
        """
        Initializes the instance of class bertviz.
        
        Args:
            modelVersion (str): The specific model of bert.
            bertModel (transformers): Fine tuned model of bert.
        Returns:
            
        """
         warnings.warn("Use only in Jupyter Environment! and donot pass device assigned bert!", ImportWarning)
         self.modelType = 'bert'
         self.modelVersion = modelVersion
         self.bertModel = bertModel
         print(f"Loading model's {self.modelVersion} Version to CPU")
         self.device =  torch.device('cpu')
         self.bertModel.to(self.device)
         self.tokenizer = BertTokenizer.from_pretrained(self.modelVersion)
         self.attention = None
         self.inputs = None
         self.input_ids = None
         self.token_type_ids = None
         self.tokens = None
         self.input_id_list = None
         self.sentence_b_start = None
         



     def loadSentencesandBertprocess(self,sentenceA=None,sentenceB=None)
             """
              This function loads sentences and prepares tokens and other
              parameters needed for bertviz.
              
             Args:
                 sentenceA (str): The first sentence.
                 sentenceB (str): The second sentence.
             
             Returns:
                
             """
         if sentenceB == None:
            self.inputs = self.tokenizer.encoder_plus(sentenceA,sentenceB, return_tensors='pt'
         else if sentenceA != None and sentenceB !=None:
             self.inputs= self.tokenizer.encoder_plus(sentenceA,sentenceB, return_tensors = 'pt')
         else:
             print("No sentences give as paramteres! Exiting!")
             return
        
         self.input_ids = self.inputs['input_ids']
         self.token_type_ids = self.inputs['token_type_ids']
         self.input_id_list = self.input_ids[0].tolist()
         if sentenceB=None: 
            self.sentence_b_start = len(self.token_type_ids[0].tolist())-1
         else:
            self.sentence_b_start = self.token_type_ids[0].tolist().index(1)
         self.attention = self.bertModel(self.input_ids, token_type_ids=self.token_type_ids)[-1]
    

     def displayBertVizVariables(self):
         """
         This function displays all variables related to bertviz
         Args:
         
         Returns:
         """
         print(f"bundled input {self.inputs}"}
         print(f"Input IDs {self.input_ids}")
         print(f"Input ID list {self.input_id_list}")
         print(f"Tokens {self.tokens}")
         print(f"Token Type IDs {self.token_type_ids}")
         print(f"Start position index of Sentence B in inputs {self.sentence_b_start}")

     def modelView(self):
         """
          This function checks all the parameters and
          displays the entire model weights for sentenceA
          and sentenceB for a fine tuned bert.
          
         Args:
         
         Returns:
            
         """
         if self.attention == None or self.inputs == None or self.input_ids == None 
         or self.input_id_list == None or self.sentence_b_start ==None or self.token_type_ids == None
         or self. tokens == None:
              print("Uninitializaed paramters! Exiting!")
              return
         model_view(self.attention, self.tokens, self.sentence_b_start)


    
     def headView(self):
         """
          This function checks all the parameters and
          displays the entire specific weights for sentenceA
          and sentenceB for a finetuned bert.
          
         Args:
         
         Returns:
            
         """
         if self.attention == None or self.inputs == None or self.input_ids == None 
         or self.input_id_list == None or self.sentence_b_start ==None or self.token_type_ids == None
         or self. tokens == None:
              print("Uninitializaed paramters! Exiting!")
              return
         head_view(self.attention, self.tokens, self.sentence_b_start)         
         

     def neuronView(self,sentenceA,sentenceB,layerNum,headNum):
         """
          This function displays the specific weights for sentences
          of a specific transformer layer of fine tuned bert
          
         Args:
              sentenceA (str): The first sentence.
              sentenceB (str): The second sentence.
              layerNum  (int): Layer number
              headNum   (int): Head number
         Returns:
            
         """
         neuron_view.show(self.bertModel, self.modelType, self.tokenizer, sentence_A, sentence_B, layer=layerNum, head=headNum)




