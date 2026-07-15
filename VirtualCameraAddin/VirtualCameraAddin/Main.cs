using ABB.Robotics.RobotStudio;
using ABB.Robotics.RobotStudio.Environment;
using System;
using System.Drawing;
using System.Windows.Forms; // [!] Necesario para las posiciones de anclaje (Dock)

namespace VirtualCameraAddin
{
    public class Main
    {
        public static void AddinMain()
        {
            RibbonTab pestanaCamera = new RibbonTab("Camera_Tab", "Camera");
            RibbonGroup grupoVision = new RibbonGroup("Vision_Group", "Control");

            CommandBarButton btnAbrir = new CommandBarButton("btnAbrirCamara", "Abrir Cámara");
            btnAbrir.HelpText = "Abre el panel de control de la cámara virtual";
            btnAbrir.DefaultEnabled = true;

            // Cargar el icono
            try
            {
                string addonDir = System.IO.Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location);
                string iconPath = System.IO.Path.Combine(addonDir, "webcam-icon-32px.png");
                if (System.IO.File.Exists(iconPath))
                {
                    btnAbrir.Image = Image.FromFile(iconPath);
                }
            }
            catch { }

            // Asignamos el evento de clic
            btnAbrir.ExecuteCommand += LevantarPestanaCamara;

            grupoVision.Controls.Add(btnAbrir);
            pestanaCamera.Groups.Add(grupoVision);
            UIEnvironment.RibbonTabs.Add(pestanaCamera);
        }

        private static void LevantarPestanaCamara(object sender, ExecuteCommandEventArgs e)
        {
            // 1. ELIMINAR EL FANTASMA
            ABB.Robotics.RobotStudio.Environment.Window ventanaFantasma = null;
            foreach (ABB.Robotics.RobotStudio.Environment.Window w in UIEnvironment.Windows)
            {
                if (w.Id == "VirtualCameraWindow")
                {
                    ventanaFantasma = w;
                    break;
                }
            }

            if (ventanaFantasma != null)
            {
                try { UIEnvironment.Windows.Remove(ventanaFantasma); } catch { }
            }

            // 2. CREAR NUEVA
            VirtualCameraControl controlCamara = new VirtualCameraControl();
            ToolWindow cameraWindow = new ToolWindow("VirtualCameraWindow", controlCamara, "Captura Cámara Virtual");

            // [!] EL PARCHE: Cuando el usuario pulse la 'X', detenemos el temporizador al instante
            cameraWindow.Closed += (s, args) =>
            {
                controlCamara.ForzarDetencion();
            };

            // 3. ANCLAJE FORZADO A LA IZQUIERDA
            UIEnvironment.Windows.AddDockedOrTabbed(cameraWindow, DockStyle.Left);
        }
    }
}