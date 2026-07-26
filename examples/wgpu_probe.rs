//! Probe WebGPU availability: enumerate adapters, create a device, run a
//! trivial compute dispatch, and read the result back.

fn main() {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    println!("== wgpu adapters ==");
    for adapter in instance.enumerate_adapters(wgpu::Backends::all()) {
        let info = adapter.get_info();
        println!(
            "  {:?} | {} | {:?}",
            info.backend, info.name, info.device_type
        );
    }

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("no wgpu adapter found");
    let info = adapter.get_info();
    println!("selected: {:?} | {}", info.backend, info.name);

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("probe"),
        ..Default::default()
    }))
    .expect("failed to create wgpu device");

    // Trivial compute: double 4 floats.
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(
            r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                if (gid.x < arrayLength(&data)) { data[gid.x] = data[gid.x] * 2.0; }
            }
            "#
            .into(),
        ),
    });
    let input: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let size = (input.len() * 4) as u64;
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buf, 0, bytemuck::cast_slice(&input));
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buf.as_entire_binding(),
        }],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&buf, 0, &staging, 0, size);
    queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
    device.poll(wgpu::PollType::Wait).expect("poll failed");
    let out: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
    println!("compute result: {:?} (expected [2, 4, 6, 8])", out);
    assert_eq!(out, vec![2.0, 4.0, 6.0, 8.0]);
    println!("WebGPU OK");
}
