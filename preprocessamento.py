# Bibliotecas mais comuns
import openpyxl

# Bibliotecas para leitura e processamentos dos dados
import pandas as pd
import numpy as np

# Bibliotecas para Processamento RMN
from sklearn import preprocessing
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder





def TratamentoDadosNiumag (diretorio_pasta, arquivo_niumag, inicio_conversao, pontos_inversao,
                           Pasta_Salvamento = None, Salvar = False, Data = None, T2_niumag_gm = False,
                           T2_niumag_av = False, T2nmr = False):

    """
    Esta função trata o dados brutos de RMN que o software da Niumag exporta retornando um pandas.DataFrame com as informações
    relevantes e com aspecto mais legível.

    Args:
        diretorio_pasta (str): O caminho do diretório onde está o arquivo excel exportado pelo software Niumag.
        arquivo_niumag (str.xlsx): Nome do arquivo que está dentro do diretório e que é um arquivo tipo Excel.
        inicio_conversao (int): Primeiro ponto de aquisição dos dados no arquivo Excel.
        pontos_inversao (int): Quantos pontos da inversão foram usados.
        Pasta_Salvamento (str): Diretório onde será salvo o arquivo gerado.
        Salvar (bool): Caso o usuário queira queira salvar esse procsssamento. 
        Data (str): Data do dia em que foi criado o arquivo.
        T2_niumag_gm (bool): Caso o usuário queira retornar o processamento da média geométrica T2.
        T2_niumag_av (bool): Caso o usuário queira retornar o processamento da média T2.
        T2_nmr (bool): Caso o usuário queira retornar os valores do Sinal e Tempo de Relaxação e não apenas os tempos de Distribuição e a Distribuição de Tamanho de poros.

    Returns:
        pandas.DataFrame: Retorna um DataFrame com dados processados do Excel exportado pelo Software Niumag. 

    Exemplos de Uso:
        Caso o usuário tenha um arquivo no estilo exportado pela Niumag que informa a amostra (no nome da amostra informa o nome do poço) essa função retornará os dados
        já específico para o processamento necessário.
    """

    niumag = str(diretorio_pasta) + str(arquivo_niumag)                           # Pasta do arquivp
    dados_niumag = pd.read_excel(niumag).drop('File Name', axis=1)                # Dataframe dos Dados da Niumag

    inicio = inicio_conversao-2                                                   # Linha que se inicia os dados de da inversão
    final = inicio+pontos_inversao                                                # Linha final da Inversão

    amostras = []
    tempo_relaxacao = []
    amplitude_sinal = []
    tempo_distribuicao = []
    distribuicao_t2 = []
    t2gm_niumag = []
    t2av_niumag = []
    fitting_erro = []
    fracao_argila = []
    poço = []
    amplitudes = []

    for i in np.arange(int(len(dados_niumag.columns)/7)):
        df = dados_niumag.T.reset_index().drop('index', axis = 1).T

        nome = dados_niumag.columns[i*7][:9]
        time = df[i*7][13:]
        sinal = df[i*7+2][13:]
        tempo = df[i*7+3][inicio:final]
        distribuicao = df[i*7+4][inicio:final]
        gm = float(df[i*7+2][1][7:-4])
        av = float(df[i*7+2][2][7:-4])
        fit_erro = float(df[i*7][0][-5:])
        argila = sum(distribuicao[:53])/sum(distribuicao)
        p = nome[:4]
        amp = df[i*7+2][5:10].fillna(0).nlargest(3)

        sinal_min = np.min(sinal)
        sinal_max = np.max(sinal)
        sinal_normalizado = (sinal - sinal_min) / (sinal_max - sinal_min)

        amostras.append(nome)
        poço.append(p)
        tempo_relaxacao.append(list(time))
        amplitude_sinal.append(list(sinal_normalizado))
        tempo_distribuicao.append(list(tempo))
        distribuicao_t2.append(list(distribuicao))
        t2gm_niumag.append(gm)
        t2av_niumag.append(av)
        fitting_erro.append(fit_erro)
        fracao_argila.append(argila)
        amplitudes.append(amp)

    dados = pd.DataFrame({'Amostra': amostras})
    dados['Poço'] = poço

    if T2_nmr == True:
      dados['Tempo Relaxacao'] = pd.Series(tempo_relaxacao)
      dados['Amplitude Relaxacao'] = pd.Series(amplitude_sinal)
      
    dados['Tempo Distribuicao'] = pd.Series(tempo_distribuicao)
    dados['Distribuicao T2'] = pd.Series(distribuicao_t2)
    dados['Fracao Argila'] =  fracao_argila
    dados['Fitting Error'] = fitting_erro
    dados['Amplitude'] = amplitudes

    if T2_niumag_gm == True:
        dados['T2 Geometrico Niumag'] = t2gm_niumag

    if T2_niumag_av == True:
        dados['T2 Medio Niumag'] = t2av_niumag

    dados = dados.sort_values(by = 'Amostra')

    if Salvar == True:
        local_salvamento = Pasta_Salvamento + 'Dados_Niumag_' + arquivo_niumag[:-5] + '_' + Data + '.xlsx'
        dados.to_excel(local_salvamento, sheet_name='Dados')                          # Salvar dataframe

    return dados

def TratamentoDadosLaboratorio (diretorio_pasta, dados_niumag, arquivo_laboratorio,
                                Pasta_Salvamento = None, Salvar = False, Data = None,
                                Fracoes_T2Han = False, Fracoes_T2Ge = False, BVIFFI = False,
                                Amplitude = False, Dados_porosidade_Transverso = False):
    """
    Esta função trata mescla os dados já processados de RMN (como processado pela função anterior e que tenha informações da distribuição de tamanho de poros)
    com os dados laboratoriais, que contenham dados de porosidade a gás e de RMN, permeabilidade a gas, litofácies das amostras.

    Args:
        diretorio_pasta (str): O caminho do diretório onde está o arquivo excel exportado pelo software Niumag.
        dados_niumag (pandas.DataFrame): DataFrame com as informações selecionadas da distribuição de tamanho de poros.
        Pasta_Salvamento (str): Diretório onde será salvo o arquivo gerado.
        Salvar (bool): Caso o usuário queira queira salvar esse procsssamento. 
        Data (str): Data do dia em que foi criado o arquivo.
        Fracoes_T2Han (bool): Caso o usuário queira retornar as frações da modelagem proposta por Han et al (2018).
        Fracoes_T2Ge (bool): Caso o usuário queira retornar as frações da modelagem proposta por Ge et al (2017).
        BVIFFI (bool): Caso o usuário queira retornar as frações da modelagem proposta por Coates et al (1999)
        Amplitude (bool): Caso o usuário queira transformar a lista com os valores da Amplitude do sinal de Relaxação em colunas
        Dados_porosidade_Transverso (bool): Caso o usuário queira transformar a lista com os valores da Distribuição de Tamanho de poros em colunas.

    Returns:
        pandas.DataFrame: Retorna um DataFrame com dados processados do Excel exportado pelo usuário com os dados do laboratório mesclado com os dados da Niumag. 

    Exemplos de Uso:
        Caso o usuário tenha um arquivo .xlsx com dados de laboratório e um pandas.DataFrame com dados de Distribuição de Tamanho de Poros
        essa função retornará os dados mesclados e prontos para regressões ou visualizações no formato pandas.DataFrame.
    """
                                  
                                  
    laboratorio = str(diretorio_pasta) + str(arquivo_laboratorio)
    dados_niumag = dados_niumag
    dados_lab = pd.read_excel(laboratorio)


    amostras = dados_niumag['Amostra']
    tempo_distribuicao = dados_niumag['Tempo Distribuicao']
    distribuicao_t2 = dados_niumag['Distribuicao T2']
    fitting_erro = dados_niumag['Fitting Error']
    fracao_argila = dados_niumag['Fracao Argila']
    porosidade_i = []
    poço = dados_niumag['Poço']
    media_ponderada_log = []
    s1h = []
    s2h = []
    s3h = []
    s4h = []
    s1g = []
    s3g = []
    s4g = []
    BVI = []
    FFI = []
    A1 = []
    A2 = []
    A3 = []

    codi_lab = preprocessing.LabelEncoder()
    categoria_lito = codi_lab.fit_transform(dados_lab['Litofacies'])
    onehot = OneHotEncoder()
    ohe = pd.DataFrame(onehot.fit_transform(dados_lab[['Litofacies']]).toarray())
    ohe.columns = onehot.categories_


    for i in np.arange(len(distribuicao_t2)):
        t2_transpose = pd.DataFrame([distribuicao_t2[i]]).T
        scaler = pd.DataFrame(MaxAbsScaler().fit_transform(t2_transpose))
        scaler_sum_phi = float(dados_lab['Porosidade RMN'][i])/float(scaler.sum())
        phi_i = []
        for j in np.arange(len(scaler)):
            p = float(scaler[0][j]*scaler_sum_phi)
            phi_i.append(p)
        porosidade_i.append(list(phi_i))

    for i in np.arange(len(porosidade_i)):
        phi_i = porosidade_i[i]
        tempo_log = np.log(tempo_distribuicao[i])
        produto_porosidade_t2_log = pd.DataFrame(phi_i*tempo_log)
        sum_num = np.sum(produto_porosidade_t2_log)
        sum_den = np.sum(phi_i)
        razao_t2 = float(np.exp(sum_num/sum_den))
        media_ponderada_log.append((razao_t2))

    dados = pd.DataFrame({'Amostra': amostras})
    dados['Poço'] = poço
    dados['Litofacies'] = dados_lab['Litofacies']
    dados['Categoria Litofacies'] = categoria_lito
    dados['Bioturbiditos'] = ohe['Bioturbated']
    dados['Dolowackstone'] = ohe['Dolowackstone']
    dados['Grainstone'] = ohe['Grainstone']
    dados['Brechado'] = ohe['Brechado']
    dados['Packstone'] = ohe['Packstone']
    #dados['Artificial'] = ohe['Artificial']
    dados['Tempo Distribuicao'] = pd.Series(tempo_distribuicao)
    dados['Distribuicao T2'] = pd.Series(distribuicao_t2)
    dados['Porosidade i'] = pd.Series(porosidade_i)
    dados['Porosidade Gas'] = dados_lab['Porosidade Gas']/100
    dados['Porosidade RMN'] = dados_lab['Porosidade RMN']/100
    dados['Permeabilidade Gas'] = dados_lab['Permeabilidade Gas']
    dados['Fracao Argila'] =  fracao_argila
    dados['Fitting Error'] = fitting_erro
    dados['T2 Ponderado Log'] = media_ponderada_log


    if Fracoes_T2Han == True:
        for i in np.arange(len(porosidade_i)):
            phi_i = pd.Series(porosidade_i[i])
            porosidade = np.sum(porosidade_i[i])
            a1h = phi_i[:74].sum()
            a2h = phi_i[74:84].sum()
            a3h = phi_i[84:92].sum()
            a4h = phi_i[92:].sum()
            phimicroh = a1h/porosidade
            phimesoh  = a2h/porosidade
            phimacroh = a3h/porosidade
            phisuperh = a4h/porosidade

            if phimicroh <= 0.0001:
              phimicroh = 0.0001
            if phimesoh <= 0.0001:
              phimesoh = 0.0001
            if phimacroh <= 0.0001:
              phimacroh = 0.0001
            if phisuperh <= 0.0001:
              phisuperh = 0.0001

            s1h.append(phimicroh)
            s2h.append(phimesoh)
            s3h.append(phimacroh)
            s4h.append(phisuperh)



        dados['S1Han'] = s1h
        dados['S2Han'] = s2h
        dados['S3Han'] = s3h
        dados['S4Han'] = s4h

    if Fracoes_T2Ge == True:
        for i in np.arange(len(porosidade_i)):
            phi_i = pd.Series(porosidade_i[i])
            phimicrog = phi_i[:75].sum()
            phimesog = phi_i[75:84].sum()
            phimacrog = phi_i[84:91].sum()
            phisuperg = phi_i[91:].sum()


            if phimicrog <= 0.0001:
              phimicrog = 0.0001
            if phimacrog <= 0.0001:
              phimacrog = 0.0001
            if phisuperg <= 0.0001:
              phisuperg = 0.0001

            s1g.append(phimicrog)
            s3g.append(phimacrog)
            s4g.append(phisuperg)



        dados['S1Ge'] = s1g
        dados['S3Ge'] = s3g
        dados['S4Ge'] = s4g

    if BVIFFI == True:
        for i in np.arange(len(porosidade_i)):
            phi_i = pd.Series(porosidade_i[i])
            b = phi_i[:76].sum()
            f = phi_i[76:].sum()


            if b <= 0.0001:
              b = 0.0001
            if f <= 0.0001:
              f = 0.0001

            BVI.append(b)
            FFI.append(f)

        dados['BVI'] = BVI
        dados['FFI'] = FFI


    if Dados_porosidade_Transverso == True:
        dataframe_porosidade = dados['Porosidade i']
        array_tempo_distribuicao = dados['Tempo Distribuicao']
        array_amostras = dados ['Amostra']
        df = pd.DataFrame([[0 for col in range(len(array_tempo_distribuicao[0]))] for row in range(len(array_amostras))])
        colunas = []
        for i in range(len(array_amostras)):
            for j in np.arange(len(array_tempo_distribuicao[0])):
                por = dataframe_porosidade[i][j]
                string = 'T2 ' + str(array_tempo_distribuicao[i][j])
                colunas.append(string)
                df[j][i] = por
        df.columns = colunas[0:128]
        dados = pd.concat([dados, df], axis = 1)

    if Amplitude == True:
      for i in np.arange(len(dados_niumag['Amplitude'])):
        lista = []
        for j in np.arange(len(dados_niumag['Amplitude'][i])):
          a = str(list(dados_niumag['Amplitude'][i].reset_index().drop('index', axis = 1).T[j]))[1:-1]
          nome = "T2 " + a

          if nome == 'T2 0.0':
            lista.append(0)
          elif nome == 'T2 10000.0':
            lista.append(dados['T2 10000'][0])
          else:
            lista.append(dados[nome][0])

        if lista[0] == 0:
          lista[0] = 0.000001
        if lista[1] == 0:
          lista[1] = 0.000001
        if lista[2] == 0:
          lista[2] = 0.000001

        A1.append(lista[0])
        A2.append(lista[1])
        A3.append(lista[2])

      dados["A1"] = A1
      dados['A2'] = A2
      dados['A3']= A3


    dados = dados.sort_values(by = 'Amostra')

    if Salvar == True:
        local_salvamento = Pasta_Salvamento + 'Dados_Gerais_' + Data + '.xlsx'
        dados.to_excel(local_salvamento, sheet_name='Dados')                          # Salvar dataframe


    return dados


def ProcessamentoDadosSDR (Dataframe):

  """
    Seleciona do DataFrame informado apenas os parâmetros necessário para realizar a regressção proposta por Kenyon et al (1988).

    Args:
        Dataframe (pandas.DataFrame): DataFrame com os dados da amostra, litofácies, o valor de T2_lm, porosidade RMN e Gás, e permeabilidade Gas.

    Returns:
        pandas.DataFrame: Retorna um Dataframe menor com as informações essenciais para a regressão dos dados de permeabilidade proposta por Kenyon et al (1988).

    """
    dados = pd.DataFrame({
        'Amostra': Dataframe['Amostra'],
        'Litofacies': Dataframe['Litofacies'],
        'T2': Dataframe['T2 Ponderado Log'],
        'Porosidade RMN': Dataframe['Porosidade RMN'],
        'Porosidade Gas': Dataframe['Porosidade Gas'],
        'Permeabilidade Gas': Dataframe['Permeabilidade Gas']
        })

    return dados

def ProcessamentoDadosCoates (Dataframe):

  """
    Seleciona do DataFrame informado apenas os parâmetros necessário para realizar a regressção proposta por Coates et al (1999).

    Args:
        Dataframe (pandas.DataFrame): DataFrame com os dados da amostra, litofácies, o valor de BVI, FFI, porosidade RMN e Gás, e permeabilidade Gas.

    Returns:
        pandas.DataFrame: Retorna um Dataframe menor com as informações essenciais para a regressão dos dados de permeabilidade proposta por Coates et al (1999).

    """
    dados = pd.DataFrame({
        'Amostra': Dataframe['Amostra'],
        'Litofacies': Dataframe['Litofacies'],
        'BVI': Dataframe['BVI'],
        'FFI': Dataframe['FFI'],
        'Porosidade RMN': Dataframe['Porosidade RMN'],
        'Porosidade Gas': Dataframe['Porosidade Gas'],
        'Permeabilidade Gas': Dataframe['Permeabilidade Gas']
        })
    return dados

def ProcessamentoDadosHan (Dataframe):

  """
    Seleciona do DataFrame informado apenas os parâmetros necessário para realizar a regressção proposta por Han et al (2018).

    Args:
        Dataframe (pandas.DataFrame): DataFrame com os dados da amostra, litofácies, o valor das frações da curva no tempo de corte S1, S2, S3 e S$, porosidade RMN e Gás, e permeabilidade Gas.

    Returns:
        pandas.DataFrame: Retorna um Dataframe menor com as informações essenciais para a regressão dos dados de permeabilidade proposta por Han et al (2018).

    """
    dados = pd.DataFrame({'Amostra': Dataframe['Amostra'],
                          'Litofacies': Dataframe['Litofacies'],
                          'Permeabilidade Gas': Dataframe['Permeabilidade Gas'],
                          'Porosidade Gas': Dataframe['Porosidade Gas'],
                          'Porosidade RMN': Dataframe['Porosidade RMN'],
                          'S1Han': Dataframe['S1Han'],
                          'S2Han': Dataframe['S2Han'],
                          'S3Han': Dataframe['S3Han'],
                          'S4Han': Dataframe['S4Han']
                          }).replace(0, np.nan).dropna().reset_index().drop('index', axis = 1)

    return dados

def ProcessamentoDadosGe (Dataframe):

  """
    Seleciona do DataFrame informado apenas os parâmetros necessário para realizar a regressção proposta por Ge et al (2017).

    Args:
        Dataframe (pandas.DataFrame): DataFrame com os dados da amostra, litofácies, o valor das frações da curva no tempo de corte S1, S3 e S$, porosidade RMN e Gás, e permeabilidade Gas.

    Returns:
        pandas.DataFrame: Retorna um Dataframe menor com as informações essenciais para a regressão dos dados de permeabilidade proposta por Ge et al (2017).

    """
    dados = pd.DataFrame({'Amostra': Dataframe['Amostra'],
                          'Litofacies': Dataframe['Litofacies'],
                          'Permeabilidade Gas': Dataframe['Permeabilidade Gas'],
                          'Porosidade Gas': Dataframe['Porosidade Gas'],
                          'Porosidade RMN': Dataframe['Porosidade RMN'],
                          'S1Ge': Dataframe['S1Ge'],
                          'S3Ge': Dataframe['S3Ge'],
                          'S4Ge': Dataframe['S4Ge']
                          }).replace(0, np.nan).dropna().reset_index().drop('index', axis = 1)

    return dados


def ProcessamentoDistribuicaoTreinoTeste (Dados_Treino, Dados_Teste,
                                          Valores = ['T2 0.01',  'T2 0.011',  'T2 0.012',  'T2 0.014',  'T2 0.015',  'T2 0.017',  'T2 0.019',  'T2 0.021',  'T2 0.024',
                                          'T2 0.027',  'T2 0.03',  'T2 0.033',  'T2 0.037',  'T2 0.041',  'T2 0.046',  'T2 0.051',  'T2 0.057',  'T2 0.064',
                                          'T2 0.071',  'T2 0.079',  'T2 0.088',  'T2 0.098',  'T2 0.109',  'T2 0.122',  'T2 0.136',  'T2 0.152',  'T2 0.169',
                                          'T2 0.189',  'T2 0.21',  'T2 0.234',  'T2 0.261',  'T2 0.291',  'T2 0.325',  'T2 0.362',  'T2 0.404',  'T2 0.45',
                                          'T2 0.502',  'T2 0.56',  'T2 0.624',  'T2 0.696',  'T2 0.776',  'T2 0.865',  'T2 0.964',  'T2 1.075',  'T2 1.199',
                                          'T2 1.337',  'T2 1.49',  'T2 1.661',  'T2 1.852',  'T2 2.065',  'T2 2.303',  'T2 2.567',  'T2 2.862',  'T2 3.191',
                                          'T2 3.558',  'T2 3.967',  'T2 4.423',  'T2 4.931',  'T2 5.497',  'T2 6.129',  'T2 6.834',  'T2 7.619',  'T2 8.494',
                                          'T2 9.471',  'T2 10.559',  'T2 11.772',  'T2 13.125',  'T2 14.634',  'T2 16.315',  'T2 18.19',  'T2 20.281',  'T2 22.612',
                                          'T2 25.21',  'T2 28.107',  'T2 31.337',  'T2 34.939',  'T2 38.954',  'T2 43.431',  'T2 48.422',  'T2 53.986',  'T2 60.19',
                                          'T2 67.108',  'T2 74.82',  'T2 83.418',  'T2 93.004',  'T2 103.693',  'T2 115.609',  'T2 128.895',  'T2 143.708',  'T2 160.223',
                                          'T2 178.636',  'T2 199.165',  'T2 222.053',  'T2 247.572',  'T2 276.023',  'T2 307.744',  'T2 343.11',  'T2 382.54',  'T2 426.502',
                                          'T2 475.516',  'T2 530.163',  'T2 591.09',  'T2 659.019',  'T2 734.754',  'T2 819.192',  'T2 913.335',  'T2 1018.296',  'T2 1135.32',
                                          'T2 1265.792',  'T2 1411.258',  'T2 1573.441',  'T2 1754.262',  'T2 1955.864',  'T2 2180.633',  'T2 2431.234',  'T2 2710.634',  'T2 3022.143',
                                          'T2 3369.45',  'T2 3756.671',  'T2 4188.391',  'T2 4669.725',  'T2 5206.375',  'T2 5804.697',  'T2 6471.778',  'T2 7215.521',  'T2 8044.736',
                                          'T2 8969.245',  'T2 10000']):
  
    """
    Seleciona do DataFrame informado as colunas da distribuição de Tamanho de Poros.

    Args:
        Dataframe (pandas.DataFrame): DataFrame com os dados da amostra, litofácies, o valor das frações da curva no tempo de corte S1, S3 e S$, porosidade RMN e Gás, e permeabilidade Gas.

    Returns:
        X_treino (pandas.DataFrame): Retorna um Dataframe com as informações essenciais para a regressão dos dados de permeabilidade que utilizem a curva de distribuição de tamanho de poros.
        X_teste (pandas.DataFrame): Retorna um Dataframe com as informações essenciais para a avaliação dos dados de permeabilidade que utilizem a curva de distribuição de tamanho de poros.
        y_treino (numpy.array): Retorna um numpy.array com as informações da permeabilidade de cada amostra para treinamento e regressão.
        y_teste (numpy.array): Retorna um numpy.array com as informações da permeabilidade de cada amostra para avaliação do modelo.

    """

  X_treino = Dados_Treino[Valores]
  y_treino = np.log10(Dados_Treino['Permeabilidade Gas']*1000)
  
  X_teste = Dados_Teste[Valores]
  y_teste = np.log10(Dados_Teste['Permeabilidade Gas']*1000)
  
  return X_treino, y_treino, X_teste, y_teste

def ProcessamentoReservatorio (Dados_com_Previsao, Modelagens = ['SDR']):

  
  """
    Realiza o processamento dos dados com valores de permeabilidade(s) modelada(s) a fim de obter parâmetros petrofísicos de reservatório proposta por Soto et al (2010).

    Args:
        Dados_com_Previsao (pandas.DataFrame): DataFrame com os dados de permeabilidade a gás e porosidade a gás e RMN.
        Modelagens (list): Lista com os nomes das modelagens que necessita de avaliação dos parâmetros do reservatório.

    Returns:
        pandas.DataFrame: Retorna um acréscimo ao DataFrame com os valores dos parâmetros petrofísicos de reservatório.

  """

  k_gas = Dados_com_Previsao['Permeabilidade Gas']
  phi_gas = Dados_com_Previsao['Porosidade Gas']
  phi_rmn = Dados_com_Previsao['Porosidade RMN']

  phi_z_gas = phi_gas/(1-phi_gas)
  phi_z_rmn = phi_rmn/(1-phi_rmn)

  rqi_gas = (np.pi/100)*np.sqrt(k_gas/phi_gas)
  fzi_gas = rqi_gas/phi_z_gas

  polar_arm_gas = phi_z_gas*(np.sqrt(fzi_gas**2 + 1))
  polar_angle_gas = np.arctan(fzi_gas)

  df = pd.DataFrame({'Phi_z_Gas': phi_z_gas,
                     'Phi_z_RMN': phi_z_rmn,
                     'RQI_Gas': rqi_gas,
                     'FZI_Gas': fzi_gas,
                     'Polar_arm_Gas': polar_arm_gas,
                     'Polar_angle_Gas': polar_angle_gas})

  for i in np.arange(len(Modelagens)):
    k_mod = Dados_com_Previsao['Permeabilidade Prevista ' + Modelagens[i]]

    rqi_mod = (np.pi/100)*np.sqrt(k_mod/phi_rmn)
    fzi_mod = rqi_mod/phi_z_rmn

    polar_arm_mod = phi_z_rmn*(np.sqrt(fzi_mod**2 + 1))
    polar_angle_mod = np.arctan(fzi_mod)


    df['RQI_' + Modelagens[i]] = rqi_mod
    df['FZI_' + Modelagens[i]] = fzi_mod
    df['Polar_arm_' + Modelagens[i]] = polar_arm_mod
    df['Polar_angle_' + Modelagens[i]] = polar_angle_mod


  return pd.concat([Dados_com_Previsao, df], axis = 1)


def DadosRidgeLine(Dados, Pasta_salvamento, Nome,
                   Distribuicao = ['T2 0.01',  'T2 0.011',  'T2 0.012',  'T2 0.014',  'T2 0.015',  'T2 0.017',  'T2 0.019',  'T2 0.021',  'T2 0.024',
                   'T2 0.027',  'T2 0.03',  'T2 0.033',  'T2 0.037',  'T2 0.041',  'T2 0.046',  'T2 0.051',  'T2 0.057',  'T2 0.064', 
                   'T2 0.071',  'T2 0.079',  'T2 0.088',  'T2 0.098',  'T2 0.109',  'T2 0.122',  'T2 0.136',  'T2 0.152',  'T2 0.169',
                   'T2 0.189',  'T2 0.21',  'T2 0.234',  'T2 0.261',  'T2 0.291',  'T2 0.325',  'T2 0.362',  'T2 0.404',  'T2 0.45',
                   'T2 0.502',  'T2 0.56',  'T2 0.624',  'T2 0.696',  'T2 0.776',  'T2 0.865',  'T2 0.964',  'T2 1.075',  'T2 1.199',
                   'T2 1.337',  'T2 1.49',  'T2 1.661',  'T2 1.852',  'T2 2.065',  'T2 2.303',  'T2 2.567',  'T2 2.862',  'T2 3.191',  
                   'T2 3.558',  'T2 3.967',  'T2 4.423',  'T2 4.931',  'T2 5.497',  'T2 6.129',  'T2 6.834',  'T2 7.619',  'T2 8.494', 
                   'T2 9.471',  'T2 10.559',  'T2 11.772',  'T2 13.125',  'T2 14.634',  'T2 16.315',  'T2 18.19',  'T2 20.281',  'T2 22.612', 
                   'T2 25.21',  'T2 28.107',  'T2 31.337',  'T2 34.939',  'T2 38.954',  'T2 43.431',  'T2 48.422',  'T2 53.986',  'T2 60.19',
                   'T2 67.108',  'T2 74.82',  'T2 83.418',  'T2 93.004',  'T2 103.693',  'T2 115.609',  'T2 128.895',  'T2 143.708',  'T2 160.223',
                   'T2 178.636',  'T2 199.165',  'T2 222.053',  'T2 247.572',  'T2 276.023',  'T2 307.744',  'T2 343.11',  'T2 382.54',  'T2 426.502',
                   'T2 475.516',  'T2 530.163',  'T2 591.09',  'T2 659.019',  'T2 734.754',  'T2 819.192',  'T2 913.335',  'T2 1018.296',  'T2 1135.32',
                   'T2 1265.792',  'T2 1411.258',  'T2 1573.441',  'T2 1754.262',  'T2 1955.864',  'T2 2180.633',  'T2 2431.234',  'T2 2710.634',  'T2 3022.143',
                   'T2 3369.45',  'T2 3756.671',  'T2 4188.391',  'T2 4669.725',  'T2 5206.375',  'T2 5804.697',  'T2 6471.778',  'T2 7215.521',  'T2 8044.736',
                   'T2 8969.245',  'T2 10000']):

  """
    Cria um DataFrame concatenado em uma única lista para todos os valores de tempo e da distribuição T2 para visualização da distribuição no formato RidgeLine.

    Args:
        Dados (pandas.DataFrame): DataFrame com os dados de distribuição T2 jpa formato em colunas.
        Pasta_Salvamento (str): Diretório onde será salvo o arquivo gerado.
        Nome (str): Nome que deseja salvar esse DataFrame.
        Distribuicao (list): Lista das colunas da distribuição T2

    Returns:
        pandas.DataFrame: Retorna um DataFrame com os valores em lista das distribuições T2 e os tempos respectivos.

  """

    porosidade_i = []
    tempo_distribuicao = []
    for i in np.arange(len(Dados)):
        phi_i = []
        tempo = []
        for j in np.arange(len(Distribuicao)):
            phi_i.append(dados[Distribuicao[j]][i])
            tempo.append(float(Distribuicao[j][3:]))
        porosidade_i.append(phi_i)
        tempo_distribuicao.append(tempo)
    
    Dados['Porosidade i'] = porosidade_i
    Dados['Tempo Distribuicao'] = tempo_distribuicao
    
    lista_tempo = []
    lista_amostra = []
    lista_t2 = []
    lista_litofacie = []
    lista_poço = []
    for i in np.arange(len(dados)):
        for j in np.arange(len(Dados['Tempo Distribuicao'][0])):
            lista_amostra.append(Dados['Amostra'][i])
            lista_tempo.append(Dados['Tempo Distribuicao'][i][j])
            lista_t2.append(Dados['Porosidade i'][i][j])
            lista_litofacie.append(Dados['Categoria Litofacies'][i])
            lista_poço.append(Dados['Poço'][i])

    df = pd.DataFrame({'Amostra': lista_amostra,
                       'Poço': lista_poço,
                       'Tempo': lista_tempo,
                       'T2': lista_t2,
                       'Litofacie': lista_litofacie})
    local_salvamento = Pasta_salvamento + 'Dados_RidgeLine_' + str(Nome) + '.xlsx'
    df.to_excel(local_salvamento, sheet_name='Dados')                          # Salvar dataframe
    
    return df
