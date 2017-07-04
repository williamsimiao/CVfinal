/*
* Codigo para deteccao de faces foi baseado do exemplo diponivel em:
* http://docs.opencv.org/2.4/doc/tutorials/objdetect/cascade_classifier/cascade_classifier.html
*/
/*Autoria:  William Simiao  - 13/0138002
            Ricardo Rachaus - 14/0161244
*/

#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/*
* Estrutura utilizada para segmentacao do frame de forma a facilitar a determinacao
* de regioes do frame sem rostos
*/
struct coluna {
  bool temRosto;
  int centroX;
};

/* Variaveis Globais */
String face_cascade_name;
CascadeClassifier face_cascade;
String window_name = "CVfinal";

//PARAMETROS DE MAIOR INFLUENCIA NA EFIENCIA
int const colsNumber = 5;
int const framesApular = 20;

Point daEsquerda;
Point daDireita;
int colWidth;
std::vector<coluna> colunaVect;
int indiceDaprimeiraFinal;
int nConsecutivasFinal;

// Delay variables
int frameCounter = 0;
std::vector<Rect> lastFacesDetected;

/*
* Dado um retangulo, retorna o ponto central dele
*/
Point calculaCentro(Rect oneFace) {
  Point center( oneFace.x + oneFace.width/2, oneFace.y + oneFace.height/2 );
  return center;
}

/*
* Verifica quais colunas estao disponiveis, i.e, sem faces no interior, e
* desenha  o retangulo vermelho que contorna a maior sequencia dessas colunas.
*/
void calculaColunasDispoiniveis(Mat frame) {
  int nConsecutivas = 0;
  int indiceDaprimeira = 0;
  nConsecutivasFinal = 0;
  indiceDaprimeiraFinal = 0;
  bool emSequencia = false;


  for(int j = 0; j < colsNumber; j++) {
    if(!colunaVect[j].temRosto) {
      if(!emSequencia) {
        indiceDaprimeira = j;
      }
      emSequencia = true;
      nConsecutivas++;
      //Verificando
      if(nConsecutivas > nConsecutivasFinal) {
        nConsecutivasFinal = nConsecutivas;
        indiceDaprimeiraFinal = indiceDaprimeira;
      }
    }
    else{
      emSequencia = false;
      nConsecutivas = 0;
    }
  }
}

void desenhaColunasDisponiveis(Mat frame) {
  //Agora vamos desenhar um retangulo da extremidade esquerda da primeira
  //coluna ate a extremidade direita da ultima
  rectangle(frame, Point(indiceDaprimeiraFinal*colWidth, 0),
            Point((indiceDaprimeiraFinal+nConsecutivasFinal)*colWidth, frame.cols),
            Scalar( 0, 0, 255 ),
            3,
            0);
}

/*
* Inicilazacao do vetor de colunas com suas coordenadas adequadas e todos com a
* booleana temRosto iniciada como falso.
*/
void inicializaColunas(Mat frame) {
  //Largura do frame dividido pelo numero de colunas
  colunaVect.resize(colsNumber);
  colWidth = frame.cols/colsNumber;
  //O inicio sera no ponto  colWidth/2
  //Apos essa variavel mantem referencia para o centro calculado da ultima coluna
  int centroDeColuna = colWidth/2;
  for(int i = 0; i < colsNumber; i++) {
    colunaVect[i].centroX = centroDeColuna;
    centroDeColuna = centroDeColuna + colWidth;
    colunaVect[i].temRosto = false;
  }
}

/*
* Verifica se o a cordenada x do ponto central de face esta a uma distancia de
* tamanho_da_coluna/2, se sim, entao esta dentro da coluna.
*/
bool isFaceInsideColuna(coluna myColuna, Rect face) {
  //ponto central da face
  Point centroFace = calculaCentro(face);
  int diferencaExtremoDireito = abs(centroFace.x + face.width/2 - myColuna.centroX);
  int diferencaExtremoEsquerdo = abs(centroFace.x - face.width/2 - myColuna.centroX);

  if( diferencaExtremoDireito < colWidth/2 ||
      diferencaExtremoEsquerdo < colWidth/2) {
        return true;
  }
  return false;
}

/*
* Desenha as colunas azuis que segmentam todo o frame
*/
void desenhaColunas(Mat frame) {
  for(int i = 0; i < colsNumber; i++) {
    rectangle(frame, Point(colWidth*i, 0),
              Point(colWidth*(i+1), frame.cols),
              Scalar( 255, 0, 0 ),
              3,
              0);
  }
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
    std::vector<Rect> faces;
    Mat frame_gray;

    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );

    //O for a baixo reseta o atrubuto booleano 'temRosto' de todas as colunas no
    //vetor colunaVect
    for(int i = 0; i < colsNumber; i++) {
      colunaVect[i].temRosto = false;
    }
    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

        for(int j = 0; j < colsNumber; j++) {
          if(isFaceInsideColuna(colunaVect[j], faces[i])) {
            colunaVect[j].temRosto = true;
          }
        }
    }
}

/* Apenas exibe as faces da ultima computacao*/
void displayFaces(Mat frame) {
    for ( size_t i = 0; i < lastFacesDetected.size(); i++ )
    {
        Point center( lastFacesDetected[i].x + lastFacesDetected[i].width/2, lastFacesDetected[i].y + lastFacesDetected[i].height/2 );
        ellipse( frame, center, Size( lastFacesDetected[i].width/2, lastFacesDetected[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
    }
}

/* main */
int main( int argc, const char** argv )
{
    face_cascade_name = "face_cascade.xml";
    VideoCapture capture;
    Mat frame, originalFrame;
    bool modoFrameAFrame;

    //carregamento do cascade
    if( !face_cascade.load( face_cascade_name ) ){ printf("Erro no carregamento do arquivo cascade\n"); return -1; };

    //Uso da camera ou entrada de video
    if(argc > 1) {
      capture = VideoCapture(argv[1]);
    }
    else {
      printf("Para usar um video como estrada basta passar o nome do arquivo como parametro na execucao do programa\n");
      printf("Um arquivo de videos exemplo eh fornecido, 'people_walking.mp4'\n");
      capture = VideoCapture(0);
    }

    if ( ! capture.isOpened() ) { printf("Erro na abertura do stream de video\n"); return -1; }

    //leitura do preimreiro frame para inicializar as colunas
    capture.read(frame);
    inicializaColunas(frame);

     while (!frame.empty())
    {
        //identificacao da tecla precionada
        char c = (char)waitKey(10);
        if( c == 27 ) { break; }

        /*Modo frame a frame(apenas para analise e debuging)*/

        // Se estiver no modo Frame a Frame leia apenas um frame e se a tecla 'a'
        // for precionada
        if(modoFrameAFrame) {
          if( c == 'a' ) {
            capture.read(frame);
          }
        }
        //Se nao estiver no modo Frame a Frame me o stream normalmente
        else{
          capture.read(frame);
        }

        //Ativando o modo Frame a Frame
        if( c == 'a' ) {
          modoFrameAFrame = true;
        }
        //Desativando o modo Frame a Frame
        if( c == 'd') {
          modoFrameAFrame = false;
        }

        //Aplicando o casdace classifier ao frame
        detectAndDisplay( frame );

        desenhaColunas(frame);
        //Verifica se o numero de frame lidos desde a ultima computacao das colunas
        //disponiveis eh igual ao parametro framesApular
        if (frameCounter == framesApular) {
          //Calcula sequencia de colunas nao ocupadas por faces a partir de frame atual
          calculaColunasDispoiniveis(frame);
          frameCounter = 0;
        }
        //Exibe as colunas calculadas
        desenhaColunasDisponiveis(frame);
        frameCounter++;

        //Instrucoes para ativacao e dasativacao do modo Frame a Frame na janela
        if(modoFrameAFrame) {
          string text = "Pressione 'a' para ir ao proximo frame ou 'd  para desativar'";
          const char * constant = text.c_str();
          putText(frame, constant, cvPoint(30,30),
          FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);
        }
        else {
          string text = "Pressione 'a' para ativar o modo frame a frame";
          const char * constant = text.c_str();
          putText(frame, constant, cvPoint(30,30),
          FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);
        }

        imshow( window_name, frame );
    }
    return 0;
}
