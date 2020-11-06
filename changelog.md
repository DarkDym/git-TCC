# Lista TO DO do TCC

    1. Pegar as imagens a partir do Drive para a função do Eigenface;
    2. Criar a função que lê o JSON da imagem e transforma no vetor Gamma;
    3. Escrever no texto o que já foi feito até o momento e estruturar os materiais e métodos utilizados referenciando os autores necessários;
    4. Criar arquivo de autenticação para o arquivo Database.py que não pode ser anexada ao github;
    5. Organizar funções na Tela_principal.py para realizar os testes dos resultados necesários no documento.
    6. Criar função que pega da câmera o rosto e faz o cálculo dos pesos a partir da face média previamente obtida.
    7. Fazer arquivos que não serão levados ao GitHub, pois contem as credenciais do Mongo e do Drive.
    8. Fazer o merge do Sandman com o Master.
    9. Salvar o valor da face média PSI no drive e no mongo.
    10. Fazer os cálculos euclidianos das faces perante as faces de fora do treinamento.

# Changelog

## 06/11/2020

    1. Modificação dos arquivos da aplicação para que seja executada pela função principal que executa a interface do sistema, localizada no arquivo Tela_principal.py.

## 29/10/2020

    1.  Modificação das funções do arquivo Google_Drive.py, sendo colocados as funções para obter as credenciais do Drive, assim não é necessário abrir um sessão externa para a autenticação do mesmo.
    2.  Inserção das imagens PHI e dos pesos OMEGA no Drive e inserção dos IDs dos JSONs das imagens(Phi e Omega) no banco de dados (Mongo).
    3.  Refatoração do código de obtenção das Eigenfaces no arquivo Eigen.py.
    4.  Modificação do arquivo changelog.md, adicionando as modificações feitas na aplicação e no TO DO.
    5.  Tarefas feitas do TO DO 
        1.  Arrumar a função do Google Drive;
        2.  Arrumar a função do Banco de Dados para colocar o indexador das imagens;

## 28/10/2020

    1. Modificação do arquivo Eigen.py, gerando a função get_eigenface(), sendo esta a mais atualizada e funcional para a geração das Eigenfaces.
    2. Criação do arquivo Google_Drive.py, sendo este o arquivo que faz a conexão com o google drive para a obtenção das imagens do sistema.
    3. Modificação do arquivo AbreIMG.py, modificações na função saveImg() para salvar as Eigenfaces.
    4. Modificação do arquivo HaarLike.py, modificação na função cutFace() para o corte da face de teste, necessário para o teste dos omegas gerados no arquivo Eigen.py.
    5. Criação do arquivo changelog.md, contém as modificações feitas no projeto e as próximas modificações necessárias a serem feitas.
    6. Modificação do arquivo README.md, adicoção do nome da professora orientadora.
    7. Remoção da chave de autenticação do arquivo Database.py.