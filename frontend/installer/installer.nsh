!include "LogicLib.nsh"

!macro customWelcomePage
  !define MUI_WELCOMEPAGE_TITLE "Welcome to NCIForge"
  !define MUI_WELCOMEPAGE_TEXT "This installer sets up NCIForge and its private CPU computation runtime.$\r$\n$\r$\nOn systems with a supported NVIDIA GPU, you can optionally install CUDA-enabled PyTorch after the application files are copied."
  !insertmacro MUI_PAGE_WELCOME
!macroend

!macro customInstall
  ${ifNot} ${isUpdated}
    IfSilent cuda_cpu_only 0
    DetailPrint "Checking for a supported NVIDIA GPU..."
    nsExec::ExecToStack '"$SYSDIR\cmd.exe" /d /c "nvidia-smi.exe -L"'
    Pop $0
    Pop $1

    StrCmp $0 "0" cuda_gpu_found cuda_cpu_only

    cuda_gpu_found:
      MessageBox MB_YESNO|MB_ICONQUESTION \
        "NCIForge detected an NVIDIA GPU.$\r$\n$\r$\nInstall CUDA-enabled PyTorch 2.11 (CUDA 12.8) now?$\r$\n$\r$\nThis optional component requires an internet connection, downloads several gigabytes, and is installed only for your Windows account. Choose No to use the bundled CPU PyTorch runtime." \
        IDYES cuda_install IDNO cuda_declined

    cuda_install:
      DetailPrint "Installing the optional CUDA PyTorch runtime..."
      nsExec::ExecToLog '"$INSTDIR\resources\backend\runtime\python.exe" "$INSTDIR\resources\backend\install_cuda_torch.py" install'
      Pop $0
      StrCmp $0 "0" cuda_success cuda_failed

    cuda_success:
      MessageBox MB_OK|MB_ICONINFORMATION \
        "CUDA PyTorch was installed and validated successfully.$\r$\n$\r$\nNCIForge will automatically use it for GPU jobs."
      Goto cuda_done

    cuda_failed:
      MessageBox MB_OK|MB_ICONEXCLAMATION \
        "CUDA PyTorch could not be installed or validated.$\r$\n$\r$\nNCIForge installation will continue with the bundled CPU PyTorch runtime. You can retry CUDA setup later."
      Goto cuda_done

    cuda_declined:
      DetailPrint "CUDA PyTorch was declined; the bundled CPU runtime remains active."
      Goto cuda_done

    cuda_cpu_only:
      DetailPrint "No supported NVIDIA GPU was detected; using bundled CPU PyTorch."

    cuda_done:
  ${endIf}
!macroend

!macro customUnInstall
  DetailPrint "Removing the optional NCIForge CUDA runtime..."
  RMDir /r "$LOCALAPPDATA\NCIForge\runtime\cuda-site-packages"
  Delete "$LOCALAPPDATA\NCIForge\runtime\cuda-runtime.json"
  RMDir "$LOCALAPPDATA\NCIForge\runtime"
!macroend
