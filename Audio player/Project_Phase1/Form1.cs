using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.UI;
using System.Speech;
using System.Speech.Synthesis;
using Emgu.CV.VideoSurveillance;
using HandGestureRecognition.SkinDetector;
using IrrKlang;

namespace Project_Phase1
{
    public partial class Form1 : Form
    {
        Capture capwebcam = null;
        bool bmcapprocess = false;
        Image<Bgr, Byte> imgOriginal;
        Image<Gray, Byte> imgProcessed;      
        Image<Bgr, Byte> currentFrame;       
        SpeechSynthesizer ss = new SpeechSynthesizer();
        Seq<Point> hull;
        Seq<Point> filteredHull;
        Seq<MCvConvexityDefect> defects;
        MCvConvexityDefect[] defectArray;
        String gesture;       
        MCvBox2D box;
        double fingLen;      
        int superflag = 0;
        string path;
        string name;
        ISound S;
        ISoundEngine eng = new ISoundEngine();      
        Hsv hsv_min;
        Hsv hsv_max;
        Ycc YCrCb_min;
        Ycc YCrCb_max;
        IColorSkinDetector skinDetector;
        PointF cogPt;
        private int contourAxisAngle;
        MCvMoments mv;
        public Form1()
        {
            InitializeComponent();           
            hsv_min = new Hsv(0, 45, 0);
            hsv_max = new Hsv(20, 255, 255);
            YCrCb_min = new Ycc(0, 131, 80);
            YCrCb_max = new Ycc(255, 185, 135);        
            mv = new MCvMoments();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            try
            {
                capwebcam = new Capture();
            }
            catch (NullReferenceException except)
            {
                tbGesture.Text = except.Message;
                return;
            }

            Application.Idle += processFrameAndUpdateGUI;
            bmcapprocess = true;
        }

        private void Form1_FormClosed(object sender, FormClosedEventArgs e)
        {
            if (capwebcam != null)
                capwebcam.Dispose();
        }

        void processFrameAndUpdateGUI(object sender, EventArgs e)
        {
          imgOriginal = capwebcam.RetrieveBgrFrame();
          currentFrame = capwebcam.QueryFrame();
          if (imgOriginal == null) return;           
          skinDetector = new YCrCbSkinDetector();
          Image<Gray, Byte> skin = skinDetector.DetectSkin(currentFrame, YCrCb_min, YCrCb_max);     
          imgProcessed = skin.SmoothGaussian(9);
          ExtractContourAndHull(imgProcessed);
          if(defects != null)
               DrawAndComputeFingersNum();
          ibOriginal.Image = currentFrame;
        }
     
        private void ExtractContourAndHull(Image<Gray, byte> skin)
        {
            using (MemStorage storage = new MemStorage())
            {

                Contour<Point> contours = skin.FindContours(Emgu.CV.CvEnum.CHAIN_APPROX_METHOD.CV_CHAIN_APPROX_SIMPLE, Emgu.CV.CvEnum.RETR_TYPE.CV_RETR_LIST, storage);
                Contour<Point> biggestContour = null;

                Double Result1 = 0;
                Double Result2 = 0;
                while (contours != null)
                {
                    Result1 = contours.Area;
                    if (Result1 > Result2)
                    {
                        Result2 = Result1;
                        biggestContour = contours;
                    }
                    contours = contours.HNext;
                }

                if (biggestContour != null)
                {
                    //currentFrame.Draw(biggestContour, new Bgr(Color.DarkViolet), 2);
                    Contour<Point> currentContour = biggestContour.ApproxPoly(biggestContour.Perimeter * 0.0025, storage);
                    currentFrame.Draw(currentContour, new Bgr(Color.LimeGreen), 2);
                    biggestContour = currentContour;


                    hull = biggestContour.GetConvexHull(Emgu.CV.CvEnum.ORIENTATION.CV_CLOCKWISE);
                    box = biggestContour.GetMinAreaRect();
                    PointF[] points = box.GetVertices();                 
                    mv = biggestContour.GetMoments();
                    CvInvoke.cvMoments(biggestContour,ref mv, 1);
                    double m00 = CvInvoke.cvGetSpatialMoment(ref mv, 0, 0) ;
                    double m10 = CvInvoke.cvGetSpatialMoment(ref mv, 1, 0) ;
                    double m01 = CvInvoke.cvGetSpatialMoment(ref mv, 0, 1) ;
                   
                    if (m00 != 0) { // calculate center
                    int xCenter = (int) Math.Round(m10/m00)*2;  //scale = 2
                    int yCenter = (int) Math.Round(m01/m00)*2;
                    cogPt.X =xCenter;
                    cogPt.Y =yCenter; 
                    }

                    double m11 = CvInvoke.cvGetCentralMoment(ref mv, 1, 1);
                    double m20 = CvInvoke.cvGetCentralMoment(ref mv, 2, 0);
                    double m02 = CvInvoke.cvGetCentralMoment(ref mv, 0, 2);
                    contourAxisAngle = calculateTilt(m11, m20, m02);
                    
                    Point[] ps = new Point[points.Length];
                    for (int i = 0; i < points.Length; i++)
                        ps[i] = new Point((int)points[i].X, (int)points[i].Y);

                    currentFrame.DrawPolyline(hull.ToArray(), true, new Bgr(200, 125, 75), 2);
                    currentFrame.Draw(new CircleF(new PointF(box.center.X, box.center.Y), 3), new Bgr(200, 125, 75), 2);

                    filteredHull = new Seq<Point>(storage);
                    for (int i = 0; i < hull.Total; i++)
                    {
                        if (Math.Sqrt(Math.Pow(hull[i].X - hull[i + 1].X, 2) + Math.Pow(hull[i].Y - hull[i + 1].Y, 2)) > box.size.Width / 10)
                        {
                            filteredHull.Push(hull[i]);
                        }
                    }

                    defects = biggestContour.GetConvexityDefacts(storage, Emgu.CV.CvEnum.ORIENTATION.CV_CLOCKWISE);

                    defectArray = defects.ToArray();
                }
            }
        }

        private int calculateTilt(double m11, double m20, double m02)
        {
            double diff = m20-m02;
            if (diff == 0) {
                if (m11 == 0)
                    return 0;
                else if (m11 > 0)
                    return 45;
                else // m11 < 0
                    return -45;
            }
            double theta = 0.5 * Math.Atan2(2*m11, diff);
            int tilt = (int) Math.Round( 57.2957795*theta);
            if ((diff > 0) && (m11 == 0))
                    return 0;
            else if ((diff < 0) && (m11 == 0))
                    return -90;
            else if ((diff > 0) && (m11 > 0)) // 0 to 45 degrees
                    return tilt;
            else if ((diff > 0) && (m11 < 0)) //-45 to 0
                return (180 + tilt); // change to counter-clockwise angle
            else if ((diff < 0) && (m11 > 0)) // 45 to 90
                    return tilt;
            else if ((diff < 0) && (m11 < 0)) //-90 to-45
                    return (180 + tilt); // change tocounter-clockwise angle
            tbGesture.Text= "Error in moments for tilt angle";
                return 0;
                } // end of calculateTilt()


        private void DrawAndComputeFingersNum()
        {
            //int fingerNum = 0;
            int fingerNum = 0;
            fingLen = 0;
               
            #region defects drawing
            for (int i = 0; i < defects.Total; i++)
            {
                PointF startPoint = new PointF((float)defectArray[i].StartPoint.X,
                                                (float)defectArray[i].StartPoint.Y);

                PointF depthPoint = new PointF((float)defectArray[i].DepthPoint.X,
                                                (float)defectArray[i].DepthPoint.Y);

                PointF endPoint = new PointF((float)defectArray[i].EndPoint.X,
                                                (float)defectArray[i].EndPoint.Y);

                

                LineSegment2D startDepthLine = new LineSegment2D(defectArray[i].StartPoint, defectArray[i].DepthPoint);

                LineSegment2D depthEndLine = new LineSegment2D(defectArray[i].DepthPoint, defectArray[i].EndPoint);

                CircleF startCircle = new CircleF(startPoint, 5f);

                CircleF depthCircle = new CircleF(depthPoint, 5f);

                CircleF endCircle = new CircleF(endPoint, 5f);
            
                if (/*(startCircle.Center.Y < box.center.Y || depthCircle.Center.Y < box.center.Y) &&*/ (startCircle.Center.Y < depthCircle.Center.Y) && (Math.Sqrt(Math.Pow(startCircle.Center.X - depthCircle.Center.X, 2) + Math.Pow(startCircle.Center.Y - depthCircle.Center.Y, 2)) > box.size.Height / 6.5))
                {
                    fingerNum++;
                    currentFrame.Draw(startDepthLine, new Bgr(Color.Green), 2);
                    if (fingLen < Math.Sqrt(Math.Pow(startCircle.Center.X - depthCircle.Center.X, 2) + Math.Pow(startCircle.Center.Y - depthCircle.Center.Y, 2)))
                        fingLen = Math.Sqrt(Math.Pow(startCircle.Center.X - depthCircle.Center.X, 2) + Math.Pow(startCircle.Center.Y - depthCircle.Center.Y, 2));
                    //currentFrame.Draw(depthEndLine, new Bgr(Color.Magenta), 2);
                }
              
                currentFrame.Draw(startCircle, new Bgr(Color.Red), 2);
                currentFrame.Draw(depthCircle, new Bgr(Color.Yellow), 5);
                //currentFrame.Draw(endCircle, new Bgr(Color.DarkBlue), 4);
            }
            #endregion        
            if(superflag == 1)
            gestureRecog(fingerNum);

        }


        private void gestureRecog(int fingerNum)
        {   
           
            if (fingerNum == 1 && contourAxisAngle > 130 && contourAxisAngle < 165)
                gesture = tbGesture.Text = "Thank You!";
          
            else if (fingerNum == 2 && contourAxisAngle > 90 && contourAxisAngle < 125)
            {
                gesture = tbGesture.Text = "Play";
                eng.SetAllSoundsPaused(false);
            }

            else if (fingerNum == 2 && contourAxisAngle > 130 && contourAxisAngle < 165)
            {
                gesture = tbGesture.Text = "Forward";
                if (tbarProgress.Value + 1 < tbarProgress.Maximum)
                {
                    tbarProgress.Value += 1;
                    S.PlayPosition = Convert.ToUInt32(tbarProgress.Value * 1000);
                }
            }
            else if (fingerNum == 2 && contourAxisAngle > 60 && contourAxisAngle < 90)
            {
                gesture = tbGesture.Text = "Rewind";
                if (tbarProgress.Value - 1 > tbarProgress.Minimum)
                {
                    tbarProgress.Value -= 1;
                    S.PlayPosition = Convert.ToUInt32(tbarProgress.Value * 1000);
                }
            }
            else if (fingerNum == 5)
            {
                gesture = tbGesture.Text = "Pause";
                eng.SetAllSoundsPaused(true);
            }

            else if (fingerNum == 4)
            {
                gesture = tbGesture.Text = "Volume Up";
                if (tbarVolume.Value + 1  < tbarVolume.Maximum)
                {
                    //eng.SoundVolume += 5;
                    tbarVolume.Value += 1;
                }
            }

            else if (fingerNum == 3)
            {
                gesture = tbGesture.Text = "Volume Down";
                if (tbarVolume.Value - 1 > tbarVolume.Minimum)
                {
                    //eng.SoundVolume -= 1;
                    tbarVolume.Value -= 1;
                }
            }
            else
                gesture = tbGesture.Text = "None";
        }

        private void btnPause_Click(object sender, EventArgs e)
        {
            if (bmcapprocess == true)
            {
                Application.Idle -= processFrameAndUpdateGUI;
                bmcapprocess = false;
                btnPause.Text = "resume";
            }
            else
            {
                Application.Idle += processFrameAndUpdateGUI;
                bmcapprocess = true;
                btnPause.Text = "pause";
            }

        }

        private void btnChooseFile_Click(object sender, EventArgs e)
        {
            eng.StopAllSounds();
            OpenFileDialog ofd = new OpenFileDialog();
            if (ofd.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                path = ofd.FileName;
                name = ofd.SafeFileName;
                tbFileName.Text = name;
                S = eng.Play2D(path, false, false, StreamMode.AutoDetect, true);
                int length = Convert.ToInt32(S.PlayLength / 1000);
                tbarProgress.Maximum = length;
                superflag = 1;
            }
        }

        private void tbarProgress_Scroll(object sender, EventArgs e)
        {
            S.PlayPosition = Convert.ToUInt32(tbarProgress.Value * 1000);

        }

        private void timer1_Tick(object sender, EventArgs e)
        {
            float volume = (float)tbarVolume.Value / 100;
            float a = volume * 100;
            //textBox1.Text = a.ToString() + "%";

            eng.SoundVolume = volume;
            // ---------------------------------
            int pos;
            if (name != null)
            {
                pos = Convert.ToInt32(S.PlayPosition / 1000);
                tbarProgress.Value = pos;
                tbElapsed.Text = pos.ToString();
            }
            //-----------------------------------
            if (name != null)
            {
                tbFileName.Text = name;
            }
            //-----------------------------------
            if (path == null)
            {
               
                tbarProgress.Enabled = false;
            }
            else
            {
                
                tbarProgress.Enabled = true;
            }
            //------------------------------------
            
        }

        private void btnHelp_Click(object sender, EventArgs e)
        {
            if (bmcapprocess == true)
            {
                Application.Idle -= processFrameAndUpdateGUI;
                bmcapprocess = false;
                btnPause.Text = "resume";
            }

            ibOriginal.Image = new Image<Bgr, Byte>("Help audio.jpg");
        }     
    }
}
