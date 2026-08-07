// Fused token + positional embedding gather:
// out[i, c] = wte[ids[i], c] + wpe[i % seq + pos, c]  (wpe skipped if use_wpe == 0)
//
// `seq` is the length of one sequence. With a training batch the rows are
// several independent sequences stacked, and each restarts its positions —
// which is the whole of what the batch axis means to this kernel. Unbatched,
// seq == t and `i % seq` is `i`.

struct Params {
    t: u32,        // number of token rows in total
    c: u32,        // embedding dim
    pos: u32,      // position offset of each sequence's first token
    use_wpe: u32,
    // The bound wte buffer holds rows [row_start, row_end) of the full table
    // (row-chunked to respect max_storage_buffer_binding_size). Each token is
    // written by exactly the dispatch whose chunk owns its id.
    row_start: u32,
    row_end: u32,
    seq: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> ids: array<u32>;
@group(0) @binding(2) var<storage, read> wte: array<f32>;
@group(0) @binding(3) var<storage, read> wpe: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let i = (wid.y * nwg.x + wid.x) * 256u + li;
    let n = p.t * p.c;
    if (i >= n) { return; }
    let t = i / p.c;
    let ch = i % p.c;
    let id = ids[t];
    if (id < p.row_start || id >= p.row_end) { return; }
    var v = wte[(id - p.row_start) * p.c + ch];
    if (p.use_wpe == 1u) {
        v = v + wpe[(t % p.seq + p.pos) * p.c + ch];
    }
    out[i] = v;
}
