using ABB.Robotics.Math;
using ABB.Robotics.RobotStudio;
using ABB.Robotics.RobotStudio.Environment;
using ABB.Robotics.RobotStudio.Stations;
using ABB.Robotics.RobotStudio.Stations.Forms;
using System;
using System.Drawing;
using System.IO;
using System.Windows.Forms;

namespace VirtualCameraAddin
{
    public class VirtualCameraControl : UserControl
    {
        private ComboBox cmbFrames;
        private TextBox txtPath;
        private Button btnBrowse;
        private TextBox txtInterval;
        private Button btnStartStop;
        private Timer captureTimer;
        private bool isCapturing = false;
        private string finalImagePath = "";

        // Bandera para evitar saturar la consola de errores a 10 FPS
        private bool errorNotificado = false;

        public VirtualCameraControl()
        {
            InitializeComponents();
            LoadFrames();
            ConfigurarCarpetaTemporal();
        }

        private void InitializeComponents()
        {
            this.Padding = new Padding(10);
            this.AutoScroll = true;

            FlowLayoutPanel panel = new FlowLayoutPanel();
            panel.Dock = DockStyle.Fill;
            panel.FlowDirection = FlowDirection.TopDown;
            panel.WrapContents = false;

            // 1. Selector de Cámara
            panel.Controls.Add(new Label() { Text = "1. Seleccionar Base (Cámara):", Width = 200, Margin = new Padding(0, 10, 0, 5) });
            cmbFrames = new ComboBox() { Width = 200, DropDownStyle = ComboBoxStyle.DropDownList };
            panel.Controls.Add(cmbFrames);

            // 2. Carpeta Temporal
            panel.Controls.Add(new Label() { Text = "2. Carpeta de procesamiento:", Width = 200, Margin = new Padding(0, 15, 0, 5) });
            txtPath = new TextBox() { Width = 200, ReadOnly = true };
            panel.Controls.Add(txtPath);

            btnBrowse = new Button() { Text = "Cambiar ruta...", Width = 100, Margin = new Padding(0, 5, 0, 0) };
            btnBrowse.Click += BtnBrowse_Click;
            panel.Controls.Add(btnBrowse);

            // 3. Frecuencia de muestreo
            panel.Controls.Add(new Label() { Text = "3. Frecuencia de sensor (ms):", Width = 200, Margin = new Padding(0, 15, 0, 5) });
            txtInterval = new TextBox() { Width = 100, Text = "100", ReadOnly = true };
            panel.Controls.Add(txtInterval);

            // 4. Botón Iniciar
            btnStartStop = new Button() { Text = "▶ Iniciar Cámara Virtual", Width = 200, Height = 40, Margin = new Padding(0, 25, 0, 0), BackColor = Color.LightGreen };
            btnStartStop.Click += BtnStartStop_Click;
            panel.Controls.Add(btnStartStop);

            this.Controls.Add(panel);

            captureTimer = new Timer();
            captureTimer.Tick += CaptureTimer_Tick;
        }

        private void ConfigurarCarpetaTemporal()
        {
            string tempDir = Path.Combine(Path.GetTempPath(), "RobotStudio_Vision_TFG");
            if (!Directory.Exists(tempDir))
            {
                Directory.CreateDirectory(tempDir);
            }
            txtPath.Text = tempDir;
            finalImagePath = Path.Combine(tempDir, "snapshot_yolo.png");
        }

        private void LoadFrames()
        {
            cmbFrames.Items.Clear();
            Station station = Station.ActiveStation;
            if (station != null)
            {
                foreach (Frame frame in station.Frames)
                {
                    cmbFrames.Items.Add(frame.Name);
                }
                if (cmbFrames.Items.Count > 0) cmbFrames.SelectedIndex = 0;
            }
        }

        private void BtnBrowse_Click(object sender, EventArgs e)
        {
            using (FolderBrowserDialog fbd = new FolderBrowserDialog())
            {
                if (fbd.ShowDialog() == DialogResult.OK)
                {
                    txtPath.Text = fbd.SelectedPath;
                    finalImagePath = Path.Combine(fbd.SelectedPath, "snapshot_yolo.png");
                }
            }
        }

        private void BtnStartStop_Click(object sender, EventArgs e)
        {
            if (!isCapturing)
            {
                errorNotificado = false; // Reseteamos la bandera al iniciar
                captureTimer.Interval = 100; // 10 FPS
                captureTimer.Start();
                isCapturing = true;
                btnStartStop.Text = "⏸ Detener Cámara Virtual";
                btnStartStop.BackColor = Color.LightCoral;
                Logger.AddMessage(new LogMessage("Cámara Virtual Iniciada (10 FPS)"));
            }
            else
            {
                ForzarDetencion();
            }
        }

        private void CaptureTimer_Tick(object sender, EventArgs e)
        {
            if (this.Parent == null)
            {
                ForzarDetencion();
                return;
            }

            try
            {
                if (cmbFrames.SelectedItem == null) return;

                string frameName = cmbFrames.SelectedItem.ToString();
                Station station = Station.ActiveStation;
                Frame camFrame = null;
                foreach (Frame f in station.Frames) if (f.Name == frameName) { camFrame = f; break; }

                if (camFrame == null)
                {
                    if (!errorNotificado)
                    {
                        Logger.AddMessage(new LogMessage("[Cámara Virtual] OJO: El frame '" + frameName + "' no se encuentra en la estación activa."));
                        errorNotificado = true;
                    }
                    return;
                }

                if (GraphicControl.ActiveGraphicControl != null)
                {
                    GraphicControl gc = GraphicControl.ActiveGraphicControl;

                    // Extracción de vectores base
                    Matrix4 m = camFrame.Transform.GlobalMatrix;
                    Vector3 pos = m.Translation;
                    Vector3 dir = m.MultiplyVector(new Vector3(0, 0, 1));   // +Z local
                    Vector3 up = m.MultiplyVector(new Vector3(0, 1, 0));   // +Y local

                    // ===== FIX SINGULARIDAD CENITAL =====
                    double len = Math.Sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
                    double cosVertical = Math.Abs(dir.z) / len;

                    if (cosVertical > 0.9995) // A menos de ~1.8 grados de la vertical pura
                    {
                        Vector3 lateral = m.MultiplyVector(new Vector3(1, 0, 0));
                        dir = new Vector3(dir.x + lateral.x * 0.02,
                                          dir.y + lateral.y * 0.02,
                                          dir.z + lateral.z * 0.02);
                    }
                    // ====================================

                    Camera tempCam = new Camera();
                    tempCam.LookFrom = pos;
                    tempCam.LookAt = pos + dir;
                    tempCam.UpDirection = up;
                    gc.SyncCamera(tempCam, true, 0);

                    // Renderizar a 640x480
                    Bitmap currentAnimationFrame = gc.ScreenShot(640, 480);
                    if (currentAnimationFrame != null)
                    {
                        string tempFile = Path.Combine(txtPath.Text, "temp_render.png");
                        currentAnimationFrame.Save(tempFile, System.Drawing.Imaging.ImageFormat.Png);
                        currentAnimationFrame.Dispose();

                        try
                        {
                            if (File.Exists(finalImagePath)) File.Delete(finalImagePath);
                            File.Move(tempFile, finalImagePath);
                        }
                        catch { /* Ignorar bloqueos de lectura de Python (1ms) */ }

                        // Si hemos llegado aquí con éxito, la cámara funciona bien
                        errorNotificado = false;
                    }
                }
            }
            catch (Exception ex)
            {
                // Solo imprimimos el error una vez para no saturar la CPU
                if (!errorNotificado)
                {
                    Logger.AddMessage(new LogMessage("[Cámara Virtual] Error crítico de renderizado: " + ex.Message));
                    errorNotificado = true;
                }
            }
        }

        public void ForzarDetencion()
        {
            if (captureTimer != null && captureTimer.Enabled)
            {
                captureTimer.Stop();
                isCapturing = false;

                btnStartStop.Text = "▶ Iniciar Cámara Virtual";
                btnStartStop.BackColor = Color.LightGreen;

                Logger.AddMessage(new LogMessage("Cámara detenida."));
            }
        }
    }
}