use std::ops;
use rand::prelude::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

#[wasm_bindgen(start)]
pub fn main_js() -> Result<(), JsValue> {
    run();

    Ok(())
}

#[derive(Copy, Clone)]
struct Vector3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vector3 {
    pub fn new(x: f64, y: f64, z: f64) -> Vector3 {
        Vector3 { x, y, z }
    }

    pub fn random(rng: &mut rand::rngs::StdRng) -> Vector3 {
        Vector3::new(rng.gen::<f64>(), rng.gen::<f64>(), rng.gen::<f64>())
    }

    pub fn random_unit(rng: &mut rand::rngs::StdRng) -> Vector3 {
        let mut result: Vector3;
        while {
            result = Vector3::random(rng) * 2.0 - Vector3::new(1.0, 1.0, 1.0);
            result.norm() >= 1.0
        } {}
        result
    }

    pub fn norm(&self) -> f64 {
        (self.x.powf(2.0) + self.y.powf(2.0) + self.z.powf(2.0)).sqrt()
    }

    pub fn unit(&self) -> Vector3 {
        *self / self.norm()
    }
}

impl ops::Add<Vector3> for Vector3 {
    type Output = Vector3;
    fn add(self, rhs: Vector3) -> Vector3 {
        Vector3::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl ops::Sub<Vector3> for Vector3 {
    type Output = Vector3;
    fn sub(self, rhs: Vector3) -> Vector3 {
        Vector3::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl ops::Div<f64> for Vector3 {
    type Output = Vector3;
    fn div(self, rhs: f64) -> Vector3 {
        Vector3::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}

impl ops::Mul<f64> for Vector3 {
    type Output = Vector3;
    fn mul(self, rhs: f64) -> Vector3 {
        Vector3::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}

impl ops::Mul<Vector3> for Vector3 {
    type Output = Vector3;
    fn mul(self, rhs: Vector3) -> Vector3 {
        Vector3::new(self.y * rhs.z - self.z * rhs.y, self.z * rhs.x - self.x * rhs.z, self.x * rhs.y - self.y * rhs.x)
    }
}

impl ops::BitAnd<Vector3> for Vector3 {
    type Output = f64;
    fn bitand(self, rhs: Vector3) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }
}

impl ops::Neg for Vector3 {
    type Output = Vector3;
    fn neg(self) -> Vector3 {
        Vector3::new(self.x * -1.0, self.y * -1.0, self.z * -1.0)
    }
}

struct Base {
    u: Vector3,
    v: Vector3,
    w: Vector3,
}

impl Base {
    pub fn new(u: Vector3, v: Vector3, w: Vector3) -> Base {
        Base { u, v, w }
    }
}

#[derive(Copy, Clone, Debug)]
struct Color {
    red: f64,
    green: f64,
    blue: f64,
}

impl Color {
    pub fn black() -> Color {
        Color { red: 0.0, green: 0.0, blue: 0.0 }
    }

    pub fn white() -> Color {
        Color { red: 255.0, green: 255.0, blue: 255.0 }
    }

    pub fn red() -> Color {
        Color { red: 255.0, green: 0.0, blue: 0.0 }
    }

    pub fn green() -> Color {
        Color { red: 0.0, green: 255.0, blue: 0.0 }
    }

    pub fn rgba(&self) -> String {
        format!("rgba({:.0},{:.0},{:.0},1.0)", self.red, self.green, self.blue)
    }
}

impl ops::AddAssign<Color> for Color {
    fn add_assign(&mut self, rhs: Color) {
        self.red += rhs.red;
        self.green += rhs.green;
        self.blue += rhs.blue;
    }
}

impl ops::DivAssign<f64> for Color {
    fn div_assign(&mut self, rhs: f64) {
        self.red /= rhs;
        self.green /= rhs;
        self.blue /= rhs;
    }
}

impl ops::Div<f64> for Color {
    type Output = Color;
    fn div(self, rhs: f64) -> Color {
        Color { red: self.red / rhs, green: self.green / rhs, blue: self.blue / rhs }
    }
}

impl ops::Mul<f64> for Color {
    type Output = Color;
    fn mul(self, rhs: f64) -> Color {
        Color { red: self.red * rhs, green: self.green * rhs, blue: self.blue * rhs }
    }
}

impl ops::Mul<Color> for Color {
    type Output = Color;
    fn mul(self, rhs: Color) -> Color {
        Color { red: self.red * rhs.red, green: self.green * rhs.green, blue: self.blue * rhs.blue }
    }
}

struct Camera {
    pos: Vector3,
    base: Base,
}

impl Camera {
    pub fn from_base(pos: Vector3, u: Vector3, v: Vector3, w: Vector3) -> Camera {
        Camera {
            pos,
            base: Base::new(u, v, w),
        }
    }

    pub fn new(pos: Vector3, lookat: Vector3, vup: Vector3, fov: f64, aspect: f64) -> Camera {
        let height = (fov / 2.0).tan();
        let width = height * aspect;
        let w = (pos - lookat).unit();
        let u = (vup * w).unit();
        let v = w * u;

        Camera {
            pos,
            base: Base::new(w * 2.0 * width, v * 2.0 * height, pos - u * width - v * height - w),
        }
    }
}

impl Camera {
    fn ray(&self, x: f64, y: f64) -> Ray {
        Ray {
            pos: self.pos,
            dir: self.base.w + self.base.u * x + self.base.v * y - self.pos,
        }
    }
}

#[derive(Copy, Clone)]
struct Ray {
    pos: Vector3,
    dir: Vector3,
}

impl Ray {
    pub fn point(&self, t: f64) -> Vector3 {
        self.pos + (self.dir * t)
    }
}

struct Hit {
    pos: Vector3,
    norm: Vector3,
    ray: Ray,
}

trait Hittable {
    fn hit(&self, ray: Ray, min: f64, max: f64) -> Option<Hit>;
    fn material(&self) -> &Box<dyn Reflectable>;
}

struct Sphere {
    pos: Vector3,
    radius: f64,
    material: Box<dyn Reflectable>,
}

impl Hittable for Sphere {
    fn hit(&self, ray: Ray, min: f64, max: f64) -> Option<Hit> {
        // (p - o)^2 = r^2
        let local = ray.pos - self.pos;
        let a = ray.dir & ray.dir;
        let b = local & ray.dir;
        let c = (local & local) - self.radius.powf(2.0);
        let d = b.powf(2.0) - a * c;

        if d > 0.0 {
            let t = (-b - d.sqrt()) / a;
            if min < t && t < max {
                let point = ray.point(t);
                return Some(Hit {
                    pos: point,
                    norm: (point - self.pos).unit(),
                    ray,
                });
            }

            let t = (-b + d.sqrt()) / a;
            if min < t && t < max {
                let point = ray.point(t);
                return Some(Hit {
                    pos: point,
                    norm: (point - self.pos).unit(),
                    ray,
                });
            }

            None
        } else {
            None
        }
    }

    fn material(&self) -> &Box<dyn Reflectable> {
        &self.material
    }
}

trait Reflectable {
    fn reflect(&self, scnene: &Scene, rng: &mut rand::rngs::StdRng, hit: Hit, n: i32) -> Vec<Color>;
}

struct Specular {}

impl Reflectable for Specular {
    fn reflect(&self, scene: &Scene, rng: &mut rand::rngs::StdRng, hit: Hit, n: i32) -> Vec<Color> {
        if (n == 30) {
            return vec![Color::black()];
        }

        let b = hit.norm * (hit.ray.dir.unit() & hit.norm);
        let dir = (hit.ray.dir.unit() - b * 2.0).unit();

        let ray = Ray { pos: hit.pos, dir };

        vec![scene.trace(rng, ray, n) * 0.9]
    }
}

struct Scatter {
    color: Color,
}

impl Reflectable for Scatter {
    fn reflect(&self, scene: &Scene, rng: &mut rand::rngs::StdRng, hit: Hit, n: i32) -> Vec<Color> {
        if (n == 30) {
            return vec![Color::black()];
        }

        let random = Vector3::random_unit(rng);

        let ray = Ray {
            pos: hit.pos,
            dir: hit.norm + random,
        };

        vec![self.color * scene.trace(rng, ray, n) * 0.8]
    }
}

struct Glass {
    ratio: f64,
}

impl Reflectable for Glass {
    fn reflect(&self, scene: &Scene, rng: &mut rand::rngs::StdRng, hit: Hit, n: i32) -> Vec<Color> {
        let b = hit.norm * (hit.ray.dir.unit() & hit.norm);
        let reflected = (hit.ray.dir.unit() - b * 2.0).unit();

        let norm = if hit.ray.dir & hit.norm > 0.0 { -hit.norm } else { hit.norm };
        let ratio = if hit.ray.dir & hit.norm > 0.0 { self.ratio } else { 1.0 / self.ratio };
        let dir = -hit.ray.dir.unit();

        let dt = dir & norm;
        let d = 1.0 - ratio.powf(2.0) * (1.0 - dt.powf(2.0));

        let mut ray = Ray {
            pos: hit.pos,
            dir: reflected,
        };

        if d > 0.0 {
            ray.dir = (dir - hit.norm * dt) * -ratio - hit.norm * d.sqrt();
        }

        vec![scene.trace(rng, ray, n+1)]
    }
}

struct Solid {
    color: Color,
}

impl Reflectable for Solid {
    fn reflect(&self, scene: &Scene, rng: &mut rand::rngs::StdRng, hit: Hit, n: i32) -> Vec<Color> {
        vec![self.color]
    }
}

struct Scene {
    camera: Camera,
    objects: Vec<Box<dyn Hittable>>,
    samples: i32,
    min: f64,
    max: f64,
}

impl Scene {
    pub fn new(camera: Camera, samples: i32, min: f64, max: f64) -> Scene {
        Scene {
            camera,
            objects: Vec::new(),
            samples,
            min,
            max,
        }
    }

    pub fn trace(&self, rng: &mut rand::rngs::StdRng, ray: Ray, n: i32) -> Color {
        let sky = 0.5 * (ray.dir.unit().y + 1.0);
        let mut color = Color {
            red: sky * 158.0 + 97.0,
            green: sky * 158.0 + 97.0,
            blue: 255.0,
        };

        for obj in &self.objects {
            if let Some(hit) = obj.hit(ray, self.min, self.max) {
                let material = obj.material();
                // TODO integrate
                let colors = material.reflect(self, rng, hit, n+1);
                color = colors[0];
                break;
            }
        }

        color
    }

    pub fn render_to(&self, context: web_sys::CanvasRenderingContext2d, rng: &mut rand::rngs::StdRng, width: i32, height: i32) {
        for j in 0..height {
            for i in 0..width {
                let mut color = Color::black();

                for k in 0..self.samples {
                    let dx = if k == 0 { 0.0 } else { rng.gen::<f64>() };
                    let dy = if k == 0 { 0.0 } else { rng.gen::<f64>() };
                    let x = (i as f64 + dx) / width as f64;
                    let y = (j as f64 + dy) / height as f64;

                    let ray = self.camera.ray(x, y);
                    color += self.trace(rng, ray, 0);
                }

                color /= self.samples as f64;

                context.set_fill_style(&JsValue::from(color.rgba()));
                context.fill_rect(i as f64, j as f64, 1.0, 1.0);
            }
        }
    }
}

pub fn run() {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document.get_element_by_id("screen").unwrap();
    let canvas: web_sys::HtmlCanvasElement = canvas.dyn_into::<web_sys::HtmlCanvasElement>().unwrap();

    let context = canvas.get_context("2d")
        .unwrap()
        .unwrap()
        .dyn_into::<web_sys::CanvasRenderingContext2d>()
        .unwrap();

    let mut rng: rand::rngs::StdRng = rand::SeedableRng::from_seed([1; 32]);

    let mut scene = Scene::new(
        // Camera::new(
        //     Vector3::new(0.0, 0.0, 1.0),
        //     Vector3::new(0.0, 0.0, 0.0),
        //     Vector3::new(0.0, 1.0, 0.0),
        //     (40.0 as f64).to_radians(),
        //     1.3,
        // ),
        Camera::from_base(
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(0.0, 0.75, 0.0),
            Vector3::new(-0.5, -0.375, -0.5),
        ),
        10,
        0.0001,
        std::f64::INFINITY,
    );

    let sphere1 = Sphere {
        pos: Vector3::new(0.5, -0.5, -2.0),
        radius: 0.5,
        material: Box::new(Glass {
            ratio: 2.0,
        }),
    };

    let sphere2 = Sphere {
        pos: Vector3::new(-0.5, -0.5, -2.0),
        radius: 0.5,
        material: Box::new(Specular {}),
    };

    let sphere3 = Sphere {
        pos: Vector3::new(0.0, 6.0, -2.0),
        radius: 6.0,
        material: Box::new(Scatter {
            color: Color { red: 0.8, green: 0.8, blue: 0.0 },
        }),
    };

    scene.objects.push(Box::new(sphere1));
    scene.objects.push(Box::new(sphere2));
    scene.objects.push(Box::new(sphere3));

    scene.render_to(context, &mut rng, 640, 480);
}
