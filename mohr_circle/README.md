# Mohr's circle and plane stress

An interactive plane-stress calculator. Enter σx, σy and τxy and get the principal stresses, maximum shear, Mohr's circle, drawings of the stress element at any angle, a yield check (von Mises and Tresca), and the strains.

## Run it

```
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app opens in your browser at http://localhost:8501. Streamlit apps have to be started with the `streamlit` command; running `app.py` directly does nothing.

## What's in the app

- **Mohr's circle:** the circle with the original element (X, Y), the rotated element (X′, Y′) and the 2θ arc, the principal stresses, and the maximum shear. A table lists every point, and another gives the stress on every 15° rotation, with a CSV download
- **Stress elements:** four elements side by side: original, rotated by a slider angle, principal (no shear) and maximum shear
- **Failure check:** von Mises and Tresca equivalent stresses and safety factors, with the stress state drawn on the yield envelope
- **Strains:** εx, εy, γxy, the thickness strain, and the principal strains from Hooke's law
- **How it works:** the equations
- Presets: textbook example, uniaxial tension, pure shear, equal biaxial tension, thin-walled cylinder, shaft in bending and torsion

## Files

- `mohr_solver.py` the math (no UI code): transformation, principal stresses, failure criteria, strains
- `app.py` the Streamlit interface
- `test_mohr_solver.py` checks (`python test_mohr_solver.py`)

## The math

Stress on a plane rotated counter-clockwise by θ:

```
σx′  = (σx + σy)/2 + (σx − σy)/2 · cos 2θ + τxy · sin 2θ
τx′y′ = −(σx − σy)/2 · sin 2θ + τxy · cos 2θ
```

These trace a circle with centre `C = (σx + σy)/2` and radius `R = √(((σx − σy)/2)² + τxy²)`. The principal stresses are `σ1,2 = C ± R`, the maximum in-plane shear is `R`, and `tan 2θp = 2τxy / (σx − σy)`. Turning the element by θ turns the point on the circle by 2θ in the same direction.

In plane stress the third principal stress is zero, so the absolute maximum shear is half the difference between the largest and smallest of (σ1, σ2, 0). When σ1 and σ2 have the same sign this is larger than R.

Failure criteria use the principal stresses and the yield strength σy:

```
von Mises:  σvM = √(σ1² − σ1σ2 + σ2²)
Tresca:     σT  = max(|σ1 − σ2|, |σ1|, |σ2|)
safety factor = σy / σeq
```

## How it is checked

`test_mohr_solver.py` verifies, over 300 random stress states and angles:

- Principal stresses and directions match the eigenvalues and eigenvectors of the stress tensor from NumPy
- The transformation equations match rotating the tensor, S′ = Q S Qᵀ
- The invariants hold: σ1 + σ2 = σx + σy and σ1σ2 = σxσy − τxy²
- A sweep over 36,001 angles finds the same extremes as the closed-form results
- Every transformed state lies on the circle, and the circle point turns by 2θ
- Uniaxial, pure shear and equal biaxial cases match hand results
- Von Mises matches its form in σx, σy, τxy, and Tresca is never less conservative
- Strains match Hooke's law, and the principal strain directions equal the principal stress directions
- Bad input (NaN, infinity, zero yield strength, bad E or ν) is rejected

## Conventions and limits

- Normal stress is positive in tension. τxy is positive when it points up (+y) on the +x face. Angles are counter-clockwise from the x axis
- On the circle, shear is plotted positive downward so the element and the circle both turn counter-clockwise
- Plane stress only (σz = 0), linear elastic and isotropic
- Von Mises and Tresca are yield criteria for ductile metals. They are not fracture criteria for brittle materials
