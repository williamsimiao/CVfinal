/*
* Codigo para deteccao de faces foi baseado do exemplo diponivel em:
* http://docs.opencv.org/2.4/doc/tutorials/objdetect/cascade_classifier/cascade_classifier.html
*/
/*Autoria:  William Simiao  - 130138002
            Ricardo Rachaus - 140161244
*/

#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

struct coluna {
  bool temRosto;
  int centroX;
};

/** Global variables */
String face_cascade_name, eyes_cascade_name;
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
String window_name = "Capture - Face detection";
//
Point daEsquerda;
Point daDireita;
int const colsNumber = 5;
int const framesApular = 20;
int colWidth;
std::vector<coluna> colunaVect;
int indiceDaprimeiraFinal;
int nConsecutivasFinal;

// Delay variables
int frameCounter = 0;
std::vector<Rect> lastFacesDetected;

/*
* Dado um retangulo, centorna o ponto central dele
*/
Point calculaCentro(Rect oneFace) {
  Point center( oneFace.x + oneFace.width/2, oneFace.y + oneFace.height/2 );
  return center;
}

/*
* Verifica quais colunas estao disponiveis, i.e, sem faces no interior, e
* desenha  o retangulo vermelho que contorna esse conjunto de colunas
*/
void calculaColunasDispoiniveis(Mat frame) {
  int nConsecutivas = 0;
  int indiceDaprimeira = 0;
  nConsecutivasFinal = 0;
  indiceDaprimeiraFinal = 0;
  bool emSequencia = false;

  //TODO: apagar esses prints
  // printf("\n" );
  // printf("Nao tem rosoto: " );

  for(int j = 0; j < colsNumber; j++) {
    if(!colunaVect[j].temRosto) {
      //TODO: apagar esses prints
      //printf("%d\t", j);
      if(!emSequencia) {
        indiceDaprimeira = j;
      }
      emSequencia = true;
      nConsecutivas++;
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
  //TODO: apagar esses prints
  //printf("indiceDaprimeiraFinal: %d\n",indiceDaprimeiraFinal);
  //printf("nConsecutivasFinal: %d\n", nConsecutivasFinal);
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
* Desenha as colunas que segmentam todo o frame
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

    //O for a baixo reseta o atrubuto temRosto de todas as colunas
    //TODO: pensar melhor sobre essa escolha
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

/** @function Exibe as faces no frame */
void displayFaces(Mat frame) {
    for ( size_t i = 0; i < lastFacesDetected.size(); i++ )
    {
        Point center( lastFacesDetected[i].x + lastFacesDetected[i].width/2, lastFacesDetected[i].y + lastFacesDetected[i].height/2 );
        ellipse( frame, center, Size( lastFacesDetected[i].width/2, lastFacesDetected[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
    }
}

/** @function main */
int main( int argc, const char** argv )
{
    face_cascade_name = "face_cascade.xml";
    eyes_cascade_name = "eyes_cascade.xml";
    VideoCapture capture;
    Mat frame, originalFrame;

    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return -1; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading eyes cascade\n"); return -1; };

    //-- 2. Read the video stream
    // capture.open( 0 );             //camera
    capture = VideoCapture("people_walking.mp4");

    if ( ! capture.isOpened() ) { printf("--(!)Error opening video capture\n"); return -1; }

    capture.read(frame);           //video
    //frame = imread("one_person.jpg");//imagem
    inicializaColunas(frame);

     while (capture.read(frame))  //video
     //while (!frame.empty() )        //imagem
    {
        if( frame.empty() )
        {
            printf(" --(!) No captured frame -- Break!");
            break;
        }

        //-- 3. Apply the classifier to the frame
        detectAndDisplay( frame );
        desenhaColunas(frame);
        if (frameCounter == 0) {
          calculaColunasDispoiniveis(frame);
        }
        desenhaColunasDisponiveis(frame);
        imshow( window_name, frame );

        // Atualiza a cada 20 frames
        frameCounter++;
        if (frameCounter >= framesApular) {
            frameCounter = 0;
        }

        char c = (char)waitKey(10);
        if( c == 27 ) { break; }

        //Frame by frame, deve usado com video

        // string text = "Pressione 'a' para ir ao proximo frame";
        // const char * constant = text.c_str();
        // putText(frame, constant, cvPoint(30,30),
        // FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200,200,250), 1, CV_AA);
        // if( c == 'a' ) {
        //   capture.read(frame);
        // }

    }
    return 0;
}
