# Alchemy3D + Qwen3.8, una imagen CUDA para Empirio (2× RTX A6000)

Esta entrega prepara un **Dockerfile** y un workflow de **GitHub Actions** que publica una imagen OCI en GitHub Container Registry. La imagen trae herramientas CUDA, Alchemy3D, TRELLIS.2 y un servidor llama.cpp; **no incrusta modelos de decenas de GB**.

## Confirmaciones

- Empirio GPU Cloud admite **imagen Docker CUDA personalizada** y puertos publicados (blog oficial). No hace falta elegir la plantilla Web Terminal si la instancia ofrece la opción de imagen custom.
- Los scripts son una construcción experimental; **NO** se compiló aquí la imagen CUDA ni se desplegó en Empirio. GitHub Actions debe terminar en verde antes de usarla.
- GitHub Actions corre en CPU y puede compilar extensiones CUDA con nvcc y `TORCH_CUDA_ARCH_LIST=8.6`, sin GPU, pero algunas extensiones podrían requerir más recursos o acceso a GPU durante build. Un fallo de build **no se ignora**.
- Dos GPU de 48 GB no equivalen a una única GPU de 96 GB. Cada proceso fija su GPU mediante `CUDA_VISIBLE_DEVICES`.

## Crear y publicar en tu GitHub

1. Crear repositorio GitHub, p. ej. `alchemy3d-dual-a6000`.
2. Copiar TODO este paquete al root de ese repo, incluyendo `.github/workflows/publish-ghcr.yml` y hacer commit/push.
3. En GitHub, ir a **Actions → Publish dual A6000 CUDA image → Run workflow**.
4. Esperar a `Build and publish` exitoso. Imagen resultante: `ghcr.io/TU_USUARIO/TU_REPO:dual-a6000-v1.1` (en minúsculas).
5. En el paquete GHCR, hacerlo **public** si Empirio no admite credenciales para registro privado. No subir claves ni tokens al repositorio.
6. Seleccionar **Custom Docker Image** en GPU Cloud, indicar imagen, `port=8765`, y configurar la variable `LAB_TOKEN` con un token aleatorio de al menos 24 caracteres.
7. Elegir oferta **2× RTX A6000** y volumen si se desean conservar pesos tras detener la instancia.

Ejemplo de token local (NO compartir el resultado): `python -c 'import secrets;print(secrets.token_urlsafe(48))'`.

## ¿Qué instala exactamente Docker build?

- Imagen base PyTorch 2.6.0, CUDA 12.4, toolkit nvcc, paquetes Python de laboratorio.
- Checkout por commit fijado de Alchemy3D y TRELLIS.2 en `/workspace/repos/`.
- `setup.sh` oficial: librerías básicas, FlashAttention, NVDiffRast, NVDiffRec, CuMesh, FlexGEMM, Blender. O-Voxel compilado con CUDA.
- llama.cpp CUDA (`llama-server`) compilado para arquitectura Ampere `sm_86`.
- API de pruebas autenticada, scripts de descarga, merges, controles e informes.

**Limitaciones:** dependencias clonadas por `setup.sh` sin SHA específico pueden cambiar; recomendamos congelar esas referencias después de la primera compilación funcional. El build depende de internet. Las imágenes pueden ser grandes y GitHub-hosted runner puede quedar sin disco/RAM; usar un runner mayor si sucede, sin pasar a instalaciones manuales en Empirio.

## Modelo Qwen: descargar una sola vez sobre almacenamiento de la instancia

Primera descarga, desde la terminal del contenedor o un `docker exec` autorizado:

```bash
python /opt/lab/scripts/download_qwen.py
```

La ruta esperada es `/workspace/qwen/Qwen3.8-27B-UD-Q4_K_XL.gguf`. Configurar `QWEN_MODEL` con esa ruta **y reiniciar el contenedor** para encender automáticamente Qwen en GPU 1.

El modelo pesa aproximadamente 17,7 GB. Se deja fuera de la imagen para evitar repeticiones de subida y descarga en cada rebuild; lo mismo se aplica a pesos de Alchemy3D/TRELLIS.2.

Para descargar los checkpoints del laboratorio:

```bash
bash /opt/lab/scripts/lab.sh download --profile minimal --aux
bash /opt/lab/scripts/lab.sh overlay
bash /opt/lab/scripts/lab.sh preflight --strict
```

Para reproducir merges y evaluar:

```bash
bash /opt/lab/scripts/lab.sh merge --methods ties deltas trim_mean deltas_cap
bash /opt/lab/scripts/lab.sh test --methods instruct_control base_texture_control ties deltas trim_mean deltas_cap --resolution 256
bash /opt/lab/scripts/lab.sh verify
```

## Servicios y seguridad

**Una sola imagen / un solo contenedor**:

- GPU 0: API autenticada de Alchemy3D en `0.0.0.0:8765` (único puerto externo). Token `LAB_TOKEN` obligatorio.
- GPU 1: llama.cpp en `127.0.0.1:8081` (API OpenAI-compatible; accesible por procesos internos, **no públicamente**).
- **No** habilita un shell público, Docker socket, ni control arbitrario para el LLM. El servidor de laboratorio permite únicamente pruebas listadas.
- El servidor de llama.cpp no realiza automáticamente inspecciones. Para un agente autónomo aún hay que desarrollar un bucle con llamadas estructuradas y evaluación de renders.

No colocar `LAB_TOKEN`, credenciales de Empirio, HF o claves privadas en archivos del repositorio.

## Problemas conocidos

- El build CUDA completo no se ha ejecutado aquí, por lo que podrían aparecer incompatibilidades de librerías o falta de recursos del runner. **No se garantiza la imagen hasta compilarla**.
- Si GHCR queda privado, Empirio necesita autenticación de registry o visibilidad pública. GHCR no publica el paquete automáticamente.
- Docker no incluye drivers NVIDIA; Empirio debe inyectarlos con su runtime CUDA.
- Los checkpoints originales descargados se mantienen inalterados; merges se generan en otro directorio.
- El tamaño final de imagen y su tiempo de carga no pueden estimarse con precisión antes del primer build.

## Referencias

- https://empiriolabs.ai/blog/introducing-gpu-cloud
- https://docs.github.com/en/actions/tutorials/publish-packages/publish-docker-images
- https://github.com/libd1/Alchemy3D
- https://github.com/microsoft/TRELLIS.2
