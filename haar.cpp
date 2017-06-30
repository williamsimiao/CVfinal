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

/** Function Headers */
void detectAndDisplay( Mat frame );

/** Global variables */
String face_cascade_name, eyes_cascade_name;
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
String window_name = "Capture - Face detection";
//by:Will
Point daEsquerda;
Point daDireita;
int colsNumber = 5;
float colWidth;
std::vector<coluna> colunaVect;


Point pontoMedio( Point esquerdo, Point direito) {
  int x = (esquerdo.x + direito.x)/2;
  int y = (esquerdo.y + direito.y)/2;
  Point meio(x, y);
  return meio;
}

Point calculaCentro(Rect oneFace) {
  Point center( oneFace.x + oneFace.width/2, oneFace.y + oneFace.height/2 );
  return center;
}

void escolhePontosExtremos(std::vector<Rect> faces) {
  int maior;
  int menor;
  if(faces.size() != 0) {
    maior = faces[0].x + faces[0].width/2;
    menor = faces[0].x + faces[0].width/2;

    int maiorInd = 0;
    int menorInd = 0;
    for(int i = 0; i < faces.size(); i++) {
      Point center = calculaCentro(faces[i]);
      if(center.x < menor) {
        menor = center.x;
        menorInd = i;
      }
      if(center.x > maior) {
        maior = center.x;
        maiorInd = i;
      }
    }

    if(faces.size() != 0) {
      daEsquerda = calculaCentro(faces[menorInd]);
      daDireita = calculaCentro(faces[maiorInd]);
    }
  }
}

void calculaRetangulosDispoiniveisEDesenha(Mat frame) {
  int nConsecutivas = 0;
  int indiceDaprimeira = 0;
  int nConsecutivasFinal = 0;
  int indiceDaprimeiraFinal = 0;
  bool emSequencia = false;

  for(int j = 0; j < colsNumber; j++) {

    if(!colunaVect[j].temRosto) {
      if(!emSequencia) {
        indiceDaprimeira = j;
      }
      emSequencia = true;
      nConsecutivas++;
    }
    else{
      emSequencia = false;
      nConsecutivas = 0;
    }
    if(nConsecutivas > nConsecutivasFinal) {
      nConsecutivasFinal = nConsecutivas;
      indiceDaprimeiraFinal = indiceDaprimeira;
    }
  }

  //Agora vamos desenhar um retangulo da extremidade esquerda da primeira
  //coluna ate a extremidade direita da ultima
  printf("nConsecutivasFinal: %d\n", nConsecutivasFinal);
  printf("ponto um: x=%d , y=%d\n",colunaVect[indiceDaprimeiraFinal].centroX, 0);
  printf("ponto dois: x=%d , y=%d\n",colunaVect[nConsecutivasFinal-1].centroX, frame.cols );
  rectangle(frame, Point(colunaVect[indiceDaprimeiraFinal].centroX, 0),
            Point(colunaVect[nConsecutivasFinal-1].centroX, frame.cols),
            Scalar( 0, 0, 255 ),
            1,
            0);

}

void inicializaColunas(Mat frame) {
  //Largura do frame dividido pelo numero de colunas
  colunaVect.resize(colsNumber);
  colWidth = frame.cols/colsNumber;
  //O inicio sera no ponto  colWidth/2
  //Apos essa variavel mantem referencia para o centro calculado da ultima coluna
  int centroDeColuna = colWidth/2;
  for(int i = 0; i < colsNumber; i++) {
    colunaVect[i].centroX = centroDeColuna;
    centroDeColuna += colWidth;
    colunaVect[i].temRosto = false;
  }
}

bool isFaceInsideColuna(coluna myColuna, Rect face) {
  Point centro = calculaCentro(face);
  int diferenca = abs(centro.x - myColuna.centroX);
  if(diferenca < colWidth/2) return true;
  return false;
}


/** @function main */
int main( int argc, const char** argv )
{
    // CommandLineParser parser(argc, argv,
    //     "{help h||}"
    //     "{face_cascade|../../data/haarcascades/haarcascade_frontalface_alt.xml|}"
    //     "{eyes_cascade|../../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml|}");
    //
    // cout << "\nThis program demonstrates using the cv::CascadeClassifier class to detect objects (Face + eyes) in a video stream.\n"
    //         "You can use Haar or LBP features.\n\n";
    // parser.printMessage();
    //
    // face_cascade_name = parser.get<string>("face_cascade");
    // eyes_cascade_name = parser.get<string>("eyes_cascade");
    face_cascade_name = "face_cascade.xml";
    eyes_cascade_name = "eyes_cascade.xml";
    VideoCapture capture;
    Mat frame;

    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return -1; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading eyes cascade\n"); return -1; };

    //-- 2. Read the video stream
    // capture.open( 0 );
    capture = VideoCapture("people_walking.mp4");

    if ( ! capture.isOpened() ) { printf("--(!)Error opening video capture\n"); return -1; }

    inicializaColunas(frame);
    // auto primeiro = high_resolution_clock::now();
    int count;
    while ( capture.read(frame) )
    {
        if( frame.empty() )
        {
            printf(" --(!) No captured frame -- Break!");
            break;
        }

        //-- 3. Apply the classifier to the frame
        detectAndDisplay( frame );
        calculaRetangulosDispoiniveisEDesenha(frame);
        imshow( window_name, frame );
        count++;

        char c = (char)waitKey(10);
        if( c == 27 ) { break; } // escape
    }
    return 0;
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

    // //by:Will
    // printf("faces.size(): %lu\n", faces.size());
    // escolhePontosExtremos(faces);
    // printf("doido\n" );
    // Point pMedio = pontoMedio(daEsquerda, daDireita);
    // printf("x: %d , y: %d\n", pMedio.x, pMedio.y);
    // int radius = 40;
    // circle( frame, pMedio, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );


    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        ellipse( frame, center, Size( faces[i].width/2, faces[i].height/2 ), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

        for(int j = 0; j < colsNumber; j++) {
          if(isFaceInsideColuna(colunaVect[j], faces[i])) {
            colunaVect[j].temRosto = true;
            //Se presumirmos que cada cabeca so pode ocupar uma coluna pode parar
            //break;
          }
        }
        // Mat faceROI = frame_gray( faces[i] );
        // std::vector<Rect> eyes;
        //
        // //-- In each face, detect eyes
        // eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );
        //
        // for ( size_t j = 0; j < eyes.size(); j++ )
        // {
        //     Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
        //     int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
        //     circle( frame, eye_center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
        // }
    }
    //-- Show what you got
}
