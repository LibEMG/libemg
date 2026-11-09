# Right hand choreography asset

Files
- right_hand.blend: editable mesh, 17-bone rig, 17 motion actions, marked preview timeline.
- right_hand.glb: portable skinned mesh/materials with named animation clips.
- hand_manifest.json: labels, timing, Blender-local target quaternions (xyzw), preview frame locations.

Coordinate convention
Blender: distal +Y, thumb +Z, palm -X. Right hand viewed from the palm has the thumb on the viewer's left when fingers point up. Forearm is fixed; wrist rotates at the origin.
GLB is converted by Blender to glTF Y-up. Use the exported transforms unchanged. The manifest pose rotations are Blender bone-local values, NOT direct replacements for glTF joint transforms. Use GLB animation tracks for playback in both apps.

Classes (17 including rest)
rest; hand_closed; hand_open; wrist_abduction; wrist_adduction; wrist_flexion; wrist_extension; index_flex; index_extend; middle_flex; middle_extend; ring_flex; ring_extend; pinky_flex; pinky_extend; thumb_flex; thumb_extend.
Individual extension straightens only that digit; other digits remain relaxed.
Wrist targets: abduction +45 degrees toward thumb, adduction -45 toward pinky, flexion +90 toward palm, extension -90 toward dorsum. These are requested visualization extremes, not range-of-motion guidance.
Pronation/supination are not included because no target angles were specified.

Timing
30 fps. Each motion action: frame 1 rest, 10 rest, 31 target, 46 target, 67 rest, 76 rest. After zero-based export: 0, .3, 1, 1.5, 2.2, 2.5 seconds.
Pose_Library_Preview is a demonstration timeline, not a selectable motion class.
The motion name is UI text; GLB does not carry a dynamic screen overlay.

Tauri integration
Place GLB and JSON in the frontend static assets. Load GLB with Three.js GLTFLoader and create AnimationMixer for gltf.scene. Populate checkboxes from manifest.clips excluding rest. Queue checked IDs in fixed or shuffled order. Play corresponding clip once (LoopOnce), clamp until the next clip starts, and update the mixer using elapsed seconds. Each clip already returns to rest. On the mixer finished event, play the next enabled ID. Display manifest label in an HTML heading above the canvas; display Rest during rest intervals. Stop/reset the previous action before starting the next. All clips key all joints, so avoid simultaneously playing full-weight actions. Apply timeScale to alter speed.
For variable timing, sample target at 1 second and rest at 0, then interpolate local joint quaternions using shortest-arc SLERP; interpolate any translation/scale tracks too. Use a smoothstep eased phase and explicit rest, transition, hold, return, rest phases. Carry the same queue/timing rules into Python.

Dear PyGui integration
Dear PyGui displays the UI and image texture; add a separate 3D renderer with glTF skinning/animation support. Load this same GLB in that renderer, evaluate the chosen animation at the current phase, render to an offscreen RGBA buffer, and update a DPG raw or dynamic texture each GUI frame. Use normalized float RGBA in the expected buffer format, contiguous storage, and flip rows if the renderer's origin differs. A static mesh-only importer will not animate this hand: the renderer must evaluate glTF joint animation and skin matrices. Keep rendering and DPG context operations on their appropriate threads.
Create checkboxes from the manifest, a text label above add_image, and a scheduler based on a monotonic clock. Reuse the exact 2.5 second clip contract. This deliverable supplies assets and integration specifications, not complete Tauri or Dear PyGui applications.

Re-export from Blender
Select the RightHand collection's rig, skin, and five nails only.
File > Export > glTF 2.0, format GLB; Selected Objects; animations enabled; animation mode Actions; sample animations; slide action ranges to zero; skins enabled; +Y Up. Keep bone names and material names. Export all actions on the rig. Exclude the archived original scene.
Do not export OBJ/STL: these do not preserve the rig and clips.
The supplied GLB already follows this configuration.

Visual limitations
Procedurally modeled, smooth stylized anthropomorphic hand; no scan textures, tendons, or corrective pose shapes. Extreme wrist and fist poses can show soft-tissue compression artifacts. Use the Blender source for further anatomical refinement.

References
https://threejs.org/docs/pages/GLTFLoader.html
https://threejs.org/manual/pages/animation-system.html
https://dearpygui.readthedocs.io/en/latest/documentation/textures.html
