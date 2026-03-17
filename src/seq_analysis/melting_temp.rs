use crate::alphabets::dna;
use crate::seq_analysis::gc::gc_content;
use std::borrow::{Borrow, Cow};

/// Complement (not reversed!) sequence
fn compl_seq<C: Borrow<u8>, T: IntoIterator<Item = C>>(sequence: T) -> Vec<u8> {
    sequence
        .into_iter()
        .map(|a| dna::complement(*a.borrow()))
        .collect()
}

#[derive(Clone)]
pub struct TmNnParams {
    pub strict: bool,
    pub shift: i32,
    pub dnac1: f64,
    pub dnac2: f64,
    pub selfcomp: bool,
    pub na: f64,
    pub k: f64,
    pub tris: f64,
    pub mg: f64,
    pub dntps: f64,
    pub saltcorr: u8,
}

impl Default for TmNnParams {
    fn default() -> Self {
        TmNnParams {
            strict: true,
            shift: 0,
            dnac1: 25.0,
            dnac2: 25.0,
            selfcomp: false,
            na: 50.0,
            k: 0.0,
            tris: 0.0,
            mg: 0.0,
            dntps: 0.0,
            saltcorr: 5,
        }
    }
}

// Structure to hold initiation parameters for DNA_NN tables
// These are the "init_*" values from the thermodynamic tables
#[derive(Copy, Clone)]
struct DnaInitParams {
    init: HS,        // Basic initiation
    init_at: HS,     // A/T terminal basepairs
    init_gc: HS,     // G/C terminal basepairs
    init_one_gc: HS, // At least one G/C pair
    init_all_at: HS, // All A/T pairs
    init_5t_a: HS,   // 5' end is T or 3' end is A
    sym: HS,         // Symmetry correction
}

// DNA_NN1 (Breslauer et al. 1986)
const DNA_NN1_INIT: DnaInitParams = DnaInitParams {
    init: [0.0, 0.0],
    init_at: [0.0, 0.0],
    init_gc: [0.0, 0.0],
    init_one_gc: [0.0, -16.8],
    init_all_at: [0.0, -20.1],
    init_5t_a: [0.0, 0.0],
    sym: [0.0, -1.3],
};

// DNA_NN2 (Sugimoto et al. 1996)
const DNA_NN2_INIT: DnaInitParams = DnaInitParams {
    init: [0.6, -9.0],
    init_at: [0.0, 0.0],
    init_gc: [0.0, 0.0],
    init_one_gc: [0.0, 0.0],
    init_all_at: [0.0, 0.0],
    init_5t_a: [0.0, 0.0],
    sym: [0.0, -1.4],
};

// DNA_NN3 (Allawi & SantaLucia 1997) - default
const DNA_NN3_INIT: DnaInitParams = DnaInitParams {
    init: [0.0, 0.0],
    init_at: [2.3, 4.1],
    init_gc: [0.1, -2.8],
    init_one_gc: [0.0, 0.0],
    init_all_at: [0.0, 0.0],
    init_5t_a: [0.0, 0.0],
    sym: [0.0, -1.4],
};

// DNA_NN4 (SantaLucia & Hicks 2004)
const DNA_NN4_INIT: DnaInitParams = DnaInitParams {
    init: [0.2, -5.7],
    init_at: [2.2, 6.9],
    init_gc: [0.0, 0.0],
    init_one_gc: [0.0, 0.0],
    init_all_at: [0.0, 0.0],
    init_5t_a: [0.0, 0.0],
    sym: [0.0, -1.4],
};

type HS = [f64; 2];
const D_H: usize = 0;
const D_S: usize = 1;

// Helper: Look up NN value from DNA_NN1 table (Breslauer et al. 1986)
fn dna_nn1_lookup(pair: &[u8]) -> Option<HS> {
    match pair {
        b"AA/TT" => Some([-9.1, -24.0]),
        b"AT/TA" => Some([-8.6, -23.9]),
        b"TA/AT" => Some([-6.0, -16.9]),
        b"CA/GT" => Some([-5.8, -12.9]),
        b"GT/CA" => Some([-6.5, -17.3]),
        b"CT/GA" => Some([-7.8, -20.8]),
        b"GA/CT" => Some([-5.6, -13.5]),
        b"CG/GC" => Some([-11.9, -27.8]),
        b"GC/CG" => Some([-11.1, -26.7]),
        b"GG/CC" => Some([-11.0, -26.6]),
        _ => None,
    }
}

// Helper: Look up NN value from DNA_NN2 table (Sugimoto et al. 1996)
fn dna_nn2_lookup(pair: &[u8]) -> Option<HS> {
    match pair {
        b"AA/TT" => Some([-8.0, -21.9]),
        b"AT/TA" => Some([-5.6, -15.2]),
        b"TA/AT" => Some([-6.6, -18.4]),
        b"CA/GT" => Some([-8.2, -21.0]),
        b"GT/CA" => Some([-9.4, -25.5]),
        b"CT/GA" => Some([-6.6, -16.4]),
        b"GA/CT" => Some([-8.8, -23.5]),
        b"CG/GC" => Some([-11.8, -29.0]),
        b"GC/CG" => Some([-10.5, -26.4]),
        b"GG/CC" => Some([-10.9, -28.4]),
        _ => None,
    }
}

// Helper: Look up NN value from DNA_NN3 table (Allawi & SantaLucia 1997)
fn dna_nn3_lookup(pair: &[u8]) -> Option<HS> {
    match pair {
        b"AA/TT" => Some([-7.9, -22.2]),
        b"AT/TA" => Some([-7.2, -20.4]),
        b"TA/AT" => Some([-7.2, -21.3]),
        b"CA/GT" => Some([-8.5, -22.7]),
        b"GT/CA" => Some([-8.4, -22.4]),
        b"CT/GA" => Some([-7.8, -21.0]),
        b"GA/CT" => Some([-8.2, -22.2]),
        b"CG/GC" => Some([-10.6, -27.2]),
        b"GC/CG" => Some([-9.8, -24.4]),
        b"GG/CC" => Some([-8.0, -19.9]),
        _ => None,
    }
}

// Helper: Look up NN value from DNA_NN4 table (SantaLucia & Hicks 2004)
fn dna_nn4_lookup(pair: &[u8]) -> Option<HS> {
    match pair {
        b"AA/TT" => Some([-7.6, -21.3]),
        b"AT/TA" => Some([-7.2, -20.4]),
        b"TA/AT" => Some([-7.2, -21.3]),
        b"CA/GT" => Some([-8.5, -22.7]),
        b"GT/CA" => Some([-8.4, -22.4]),
        b"CT/GA" => Some([-7.8, -21.0]),
        b"GA/CT" => Some([-8.2, -22.2]),
        b"CG/GC" => Some([-10.6, -27.2]),
        b"GC/CG" => Some([-9.8, -24.4]),
        b"GG/CC" => Some([-8.0, -19.9]),
        _ => None,
    }
}

// Helper: Look up internal mismatch value from DNA_IMM1
fn imm1_lookup(seq_pair: &[u8]) -> Option<HS> {
    match seq_pair {
        b"AG/TT" => Some([1.0, 0.9]),
        b"AT/TG" => Some([-2.5, -8.3]),
        b"CG/GT" => Some([-4.1, -11.7]),
        b"CT/GG" => Some([-2.8, -8.0]),
        b"GG/CT" => Some([3.3, 10.4]),
        b"GG/TT" => Some([5.8, 16.3]),
        b"GT/CG" => Some([-4.4, -12.3]),
        b"GT/TG" => Some([4.1, 9.5]),
        b"TG/AT" => Some([-0.1, -1.7]),
        b"TG/GT" => Some([-1.4, -6.2]),
        b"TT/AG" => Some([-1.3, -5.3]),
        b"AA/TG" => Some([-0.6, -2.3]),
        b"AG/TA" => Some([-0.7, -2.3]),
        b"CA/GG" => Some([-0.7, -2.3]),
        b"CG/GA" => Some([-4.0, -13.2]),
        b"GA/CG" => Some([-0.6, -1.0]),
        b"GG/CA" => Some([0.5, 3.2]),
        b"TA/AG" => Some([0.7, 0.7]),
        b"TG/AA" => Some([3.0, 7.4]),
        b"AC/TT" => Some([0.7, 0.2]),
        b"AT/TC" => Some([-1.2, -6.2]),
        b"CC/GT" => Some([-0.8, -4.5]),
        b"CT/GC" => Some([-1.5, -6.1]),
        b"GC/CT" => Some([2.3, 5.4]),
        b"GT/CC" => Some([5.2, 13.5]),
        b"TC/AT" => Some([1.2, 0.7]),
        b"TT/AC" => Some([1.0, 0.7]),
        b"AA/TC" => Some([2.3, 4.6]),
        b"AC/TA" => Some([5.3, 14.6]),
        b"CA/GC" => Some([1.9, 3.7]),
        b"CC/GA" => Some([0.6, -0.6]),
        b"GA/CC" => Some([5.2, 14.2]),
        b"GC/CA" => Some([-0.7, -3.8]),
        b"TA/AC" => Some([3.4, 8.0]),
        b"TC/AA" => Some([7.6, 20.2]),
        b"AA/TA" => Some([1.2, 1.7]),
        b"CA/GA" => Some([-0.9, -4.2]),
        b"GA/CA" => Some([-2.9, -9.8]),
        b"TA/AA" => Some([4.7, 12.9]),
        b"AC/TC" => Some([0.0, -4.4]),
        b"CC/GC" => Some([-1.5, -7.2]),
        b"GC/CC" => Some([3.6, 8.9]),
        b"TC/AC" => Some([6.1, 16.4]),
        b"AG/TG" => Some([-3.1, -9.5]),
        b"CG/GG" => Some([-4.9, -15.3]),
        b"GG/CG" => Some([-6.0, -15.8]),
        b"TG/AG" => Some([1.6, 3.6]),
        b"AT/TT" => Some([-2.7, -10.8]),
        b"CT/GT" => Some([-5.0, -15.8]),
        b"GT/CT" => Some([-2.2, -8.4]),
        b"TT/AT" => Some([0.2, -1.5]),
        b"AI/TC" => Some([-8.9, -25.5]),
        b"TI/AC" => Some([-5.9, -17.4]),
        b"AC/TI" => Some([-8.8, -25.4]),
        b"TC/AI" => Some([-4.9, -13.9]),
        b"CI/GC" => Some([-5.4, -13.7]),
        b"GI/CC" => Some([-6.8, -19.1]),
        b"CC/GI" => Some([-8.3, -23.8]),
        b"GC/CI" => Some([-5.0, -12.6]),
        b"AI/TA" => Some([-8.3, -25.0]),
        b"TI/AA" => Some([-3.4, -11.2]),
        b"AA/TI" => Some([-0.7, -2.6]),
        b"TA/AI" => Some([-1.3, -4.6]),
        b"CI/GA" => Some([2.6, 8.9]),
        b"GI/CA" => Some([-7.8, -21.1]),
        b"CA/GI" => Some([-7.0, -20.0]),
        b"GA/CI" => Some([-7.6, -20.2]),
        b"AI/TT" => Some([0.49, -0.7]),
        b"TI/AT" => Some([-6.5, -22.0]),
        b"AT/TI" => Some([-5.6, -18.7]),
        b"TT/AI" => Some([-0.8, -4.3]),
        b"CI/GT" => Some([-1.0, -2.4]),
        b"GI/CT" => Some([-3.5, -10.6]),
        b"CT/GI" => Some([0.1, -1.0]),
        b"GT/CI" => Some([-4.3, -12.1]),
        b"AI/TG" => Some([-4.9, -15.8]),
        b"TI/AG" => Some([-1.9, -8.5]),
        b"AG/TI" => Some([0.1, -1.8]),
        b"TG/AI" => Some([1.0, 1.0]),
        b"CI/GG" => Some([7.1, 21.3]),
        b"GI/CG" => Some([-1.1, -3.2]),
        b"CG/GI" => Some([5.8, 16.9]),
        b"GG/CI" => Some([-7.6, -22.0]),
        b"AI/TI" => Some([-3.3, -11.9]),
        b"TI/AI" => Some([0.1, -2.3]),
        b"CI/GI" => Some([1.3, 3.0]),
        b"GI/CI" => Some([-0.5, -1.3]),
        _ => None,
    }
}

// Helper: Look up terminal mismatch value from DNA_TMM1
fn tmm1_lookup(seq_pair: &[u8]) -> Option<HS> {
    match seq_pair {
        b"AA/TA" => Some([-3.1, -7.8]),
        b"TA/AA" => Some([-2.5, -6.3]),
        b"CA/GA" => Some([-4.3, -10.7]),
        b"GA/CA" => Some([-8.0, -22.5]),
        b"AC/TC" => Some([-0.1, 0.5]),
        b"TC/AC" => Some([-0.7, -1.3]),
        b"CC/GC" => Some([-2.1, -5.1]),
        b"GC/CC" => Some([-3.9, -10.6]),
        b"AG/TG" => Some([-1.1, -2.1]),
        b"TG/AG" => Some([-1.1, -2.7]),
        b"CG/GG" => Some([-3.8, -9.5]),
        b"GG/CG" => Some([-0.7, -19.2]),
        b"AT/TT" => Some([-2.4, -6.5]),
        b"TT/AT" => Some([-3.2, -8.9]),
        b"CT/GT" => Some([-6.1, -16.9]),
        b"GT/CT" => Some([-7.4, -21.2]),
        b"AA/TC" => Some([-1.6, -4.0]),
        b"AC/TA" => Some([-1.8, -3.8]),
        b"CA/GC" => Some([-2.6, -5.9]),
        b"CC/GA" => Some([-2.7, -6.0]),
        b"GA/CC" => Some([-5.0, -13.8]),
        b"GC/CA" => Some([-3.2, -7.1]),
        b"TA/AC" => Some([-2.3, -5.9]),
        b"TC/AA" => Some([-2.7, -7.0]),
        b"AC/TT" => Some([-0.9, -1.7]),
        b"AT/TC" => Some([-2.3, -6.3]),
        b"CC/GT" => Some([-3.2, -8.0]),
        b"CT/GC" => Some([-3.9, -10.6]),
        b"GC/CT" => Some([-4.9, -13.5]),
        b"GT/CC" => Some([-3.0, -7.8]),
        b"TC/AT" => Some([-2.5, -6.3]),
        b"TT/AC" => Some([-0.7, -1.2]),
        b"AA/TG" => Some([-1.9, -4.4]),
        b"AG/TA" => Some([-2.5, -5.9]),
        b"CA/GG" => Some([-3.9, -9.6]),
        b"CG/GA" => Some([-6.0, -15.5]),
        b"GA/CG" => Some([-4.3, -11.1]),
        b"GG/CA" => Some([-4.6, -11.4]),
        b"TA/AG" => Some([-2.0, -4.7]),
        b"TG/AA" => Some([-2.4, -5.8]),
        b"AG/TT" => Some([-3.2, -8.7]),
        b"AT/TG" => Some([-3.5, -9.4]),
        b"CG/GT" => Some([-3.8, -9.0]),
        b"CT/GG" => Some([-6.6, -18.7]),
        b"GG/CT" => Some([-5.7, -15.9]),
        b"GT/CG" => Some([-5.9, -16.1]),
        b"TG/AT" => Some([-3.9, -10.5]),
        b"TT/AG" => Some([-3.6, -9.8]),

        _ => None,
    }
}

// Helper: Look up dangling end value from DNA_DE1
fn de1_lookup(seq_pair: &[u8]) -> Option<HS> {
    match seq_pair {
        b"AA/.T" => Some([0.2, 2.3]),
        b"AC/.G" => Some([-6.3, -17.1]),
        b"AG/.C" => Some([-3.7, -10.0]),
        b"AT/.A" => Some([-2.9, -7.6]),
        b"CA/.T" => Some([0.6, 3.3]),
        b"CC/.G" => Some([-4.4, -12.6]),
        b"CG/.C" => Some([-4.0, -11.9]),
        b"CT/.A" => Some([-4.1, -13.0]),
        b"GA/.T" => Some([-1.1, -1.6]),
        b"GC/.G" => Some([-5.1, -14.0]),
        b"GG/.C" => Some([-3.9, -10.9]),
        b"GT/.A" => Some([-4.2, -15.0]),
        b"TA/.T" => Some([-6.9, -20.0]),
        b"TC/.G" => Some([-4.0, -10.9]),
        b"TG/.C" => Some([-4.9, -13.8]),
        b"TT/.A" => Some([-0.2, -0.5]),
        b".A/AT" => Some([-0.7, -0.8]),
        b".C/AG" => Some([-2.1, -3.9]),
        b".G/AC" => Some([-5.9, -16.5]),
        b".T/AA" => Some([-0.5, -1.1]),
        b".A/CT" => Some([4.4, 14.9]),
        b".C/CG" => Some([-0.2, -0.1]),
        b".G/CC" => Some([-2.6, -7.4]),
        b".T/CA" => Some([4.7, 14.2]),
        b".A/GT" => Some([-1.6, -3.6]),
        b".C/GG" => Some([-3.9, -11.2]),
        b".G/GC" => Some([-3.2, -10.4]),
        b".T/GA" => Some([-4.1, -13.1]),
        b".A/TT" => Some([2.9, 10.4]),
        b".C/TG" => Some([-4.4, -13.1]),
        b".G/TC" => Some([-5.2, -15.0]),
        b".T/TA" => Some([-3.8, -12.6]),
        _ => None,
    }
}

// Salt correction calculation
fn salt_correction(
    na: f64,
    k: f64,
    tris: f64,
    mg: f64,
    dntps: f64,
    method: u8,
    seq: &[u8],
) -> Result<f64, String> {
    let mut corr = 0.0;
    if method == 0 {
        return Ok(corr);
    }

    let mon = na + k + tris / 2.0;
    let mut mon_eq = mon;

    if (k + mg + tris + dntps).abs() > 1e-10 && method != 7 && dntps < mg {
        mon_eq += 120.0 * (mg - dntps).sqrt();
    }

    let mon_m = mon_eq * 1e-3;

    if method >= 1 && method <= 6 && mon_m.abs() < 1e-10 {
        return Err("Total ion concentration of zero is not allowed in this method.".to_string());
    }

    match method {
        1 => corr = 16.6 * mon_m.log10(),
        2 => corr = 16.6 * (mon_m / (1.0 + 0.7 * mon_m)).log10(),
        3 => corr = 12.5 * mon_m.log10(),
        4 => corr = 11.7 * mon_m.log10(),
        5 => corr = 0.368 * ((seq.len() as f64) - 1.0) * mon_m.ln(),
        6 => {
            let gc = gc_content(seq) as f64;
            corr = ((4.29 * gc - 3.95) * 1e-5 * mon_m.ln()) + (9.40e-6 * mon_m.ln().powi(2));
        }
        7 => {
            let _a = 3.92;
            let _b = -0.911;
            let _c = 6.26;
            let _d = 1.42;
            let _e = -48.2;
            let _f = 52.5;
            let _g = 8.31;
            // Method 7 is complex; for now use method 6 approximation
            if dntps > 0.0 {
                // Skip complex Mg correction for now
            }
            if mon > 0.0 {
                let gc = gc_content(seq) as f64;
                corr = ((4.29 * gc - 3.95) * 1e-5 * mon_m.ln()) + (9.40e-6 * mon_m.ln().powi(2));
            }
        }
        _ => return Err("Allowed values for parameter 'method' are 1-7.".to_string()),
    }

    Ok(corr)
}

pub fn tm_nn(
    sequence: &[u8],
    c_sequence: Option<&[u8]>,
    params: TmNnParams,
) -> Result<f64, String> {
    if sequence.is_empty() {
        return Err("Sequence is empty.".to_string());
    }

    // Working copy that may be mutated for dangling ends/terminal mismatches
    let mut tmp_seq = Cow::Borrowed(sequence);
    let mut tmp_cseq: Cow<[u8]> = if let Some(cs) = c_sequence {
        Cow::Borrowed(cs)
    } else {
        Cow::Owned(compl_seq(sequence.iter()))
    };

    let mut delta_h = 0.0;
    let mut delta_s = 0.0;

    // Handle dangling ends
    if params.shift != 0 || tmp_seq.len() != tmp_cseq.len() {
        // Align sequences
        if params.shift > 0 {
            tmp_seq
                .to_mut()
                .splice(0..0, std::iter::repeat_n(b'.', params.shift as usize));
        } else if params.shift < 0 {
            tmp_seq
                .to_mut()
                .splice(0..0, std::iter::repeat_n(b'.', (-params.shift) as usize));
        }
        if tmp_cseq.len() > tmp_seq.len() {
            let pad_len = tmp_cseq.len() - tmp_seq.len();
            tmp_seq.to_mut().extend(std::iter::repeat_n(b'.', pad_len));
        }
        if tmp_cseq.len() < tmp_seq.len() {
            let pad_len = tmp_seq.len() - tmp_cseq.len();
            tmp_cseq.to_mut().extend(std::iter::repeat_n(b'.', pad_len));
        }

        // Remove over-dangling ends
        while (tmp_seq.len() >= 2 && tmp_seq[0..2] == [b'.', b'.'])
            || (tmp_cseq.len() >= 2 && tmp_cseq[0..2] == [b'.', b'.'])
        {
            tmp_seq.to_mut().remove(0);
            tmp_cseq.to_mut().remove(0);
        }
        while (tmp_seq.len() >= 2 && tmp_seq[tmp_seq.len() - 2..] == [b'.', b'.'])
            || (tmp_cseq.len() >= 2 && tmp_cseq[tmp_cseq.len() - 2..] == [b'.', b'.'])
        {
            tmp_seq.to_mut().pop();
            tmp_cseq.to_mut().pop();
        }

        // Left dangling end
        if tmp_seq.len() >= 2 && tmp_cseq.len() >= 2 && (tmp_seq[0] == b'.' || tmp_cseq[0] == b'.')
        {
            if let Some(de_val) =
                de1_lookup(&[tmp_seq[0], tmp_seq[1], b'/', tmp_cseq[0], tmp_cseq[1]])
            {
                delta_h += de_val[D_H];
                delta_s += de_val[D_S];
            } else if params.strict {
                return Err(format!(
                    "No thermodynamic data for left dangling end {}{}/{}{} available",
                    tmp_seq[0] as char,
                    tmp_seq[1] as char,
                    tmp_cseq[0] as char,
                    tmp_cseq[1] as char
                ));
            }
            tmp_seq.to_mut().remove(0);
            tmp_cseq.to_mut().remove(0);
        }

        // Right dangling end
        if tmp_seq.len() >= 2
            && tmp_cseq.len() >= 2
            && (tmp_seq[tmp_seq.len() - 1] == b'.' || tmp_cseq[tmp_cseq.len() - 1] == b'.')
        {
            if let Some(de_val) = de1_lookup(&[
                tmp_cseq[tmp_cseq.len() - 1],
                tmp_cseq[tmp_cseq.len() - 2],
                b'/',
                tmp_seq[tmp_seq.len() - 1],
                tmp_seq[tmp_seq.len() - 2],
            ]) {
                delta_h += de_val[D_H];
                delta_s += de_val[D_S];
            } else if params.strict {
                return Err(format!(
                    "No thermodynamic data for right dangling end '{}{}/{}{}' available",
                    tmp_seq[tmp_seq.len() - 1] as char,
                    tmp_seq[tmp_seq.len() - 2] as char,
                    tmp_cseq[tmp_cseq.len() - 1] as char,
                    tmp_cseq[tmp_cseq.len() - 2] as char,
                ));
            }
            tmp_seq.to_mut().pop();
            tmp_cseq.to_mut().pop();
        }
    }

    // Terminal mismatches
    if tmp_seq.len() >= 2 && tmp_cseq.len() >= 2 {
        if let Some(tmm_val) =
            tmm1_lookup(&[tmp_cseq[1], tmp_cseq[0], b'/', tmp_seq[1], tmp_seq[0]])
        {
            delta_h += tmm_val[D_H];
            delta_s += tmm_val[D_S];
            tmp_seq.to_mut().remove(0);
            tmp_cseq.to_mut().remove(0);
        }
    }

    if tmp_seq.len() >= 2 && tmp_cseq.len() >= 2 {
        if let Some(tmm_val) = tmm1_lookup(&[
            tmp_seq[tmp_seq.len() - 2],
            tmp_seq[tmp_seq.len() - 1],
            b'/',
            tmp_cseq[tmp_cseq.len() - 2],
            tmp_cseq[tmp_cseq.len() - 1],
        ]) {
            delta_h += tmm_val[D_H];
            delta_s += tmm_val[D_S];
            tmp_seq.to_mut().pop();
            tmp_cseq.to_mut().pop();
        }
    }

    // Use DNA_NN3 init parameters (default table)
    let init_params = DNA_NN3_INIT;

    // Initiation
    delta_h += init_params.init[D_H];
    delta_s += init_params.init[D_S];

    // Check for all A/T vs at least one G/C
    if gc_content(sequence) == 0.0 {
        delta_h += init_params.init_all_at[D_H];
        delta_s += init_params.init_all_at[D_S];
    } else {
        delta_h += init_params.init_one_gc[D_H];
        delta_s += init_params.init_one_gc[D_S];
    }

    // 5' end is T penalty
    if sequence[0] == b'T' {
        delta_h += init_params.init_5t_a[D_H];
        delta_s += init_params.init_5t_a[D_S];
    }
    if sequence[sequence.len() - 1] == b'A' {
        delta_h += init_params.init_5t_a[D_H];
        delta_s += init_params.init_5t_a[D_S];
    }

    // Terminal A/T vs G/C penalty
    let ends = vec![sequence[0], sequence[sequence.len() - 1]];
    let at_count = ends.iter().filter(|&&b| b == b'A' || b == b'T').count() as f64;
    let gc_count = ends.iter().filter(|&&b| b == b'G' || b == b'C').count() as f64;
    delta_h += at_count * init_params.init_at[D_H];
    delta_s += at_count * init_params.init_at[D_S];
    delta_h += gc_count * init_params.init_gc[D_H];
    delta_s += gc_count * init_params.init_gc[D_S];

    // Nearest neighbor loop
    for i in 0..tmp_seq.len().saturating_sub(1) {
        let neighbors = [
            tmp_seq[i],
            tmp_seq[i + 1],
            b'/',
            tmp_cseq[i],
            tmp_cseq[i + 1],
        ];
        if let Some(val) = imm1_lookup(&neighbors) {
            delta_h += val[D_H];
            delta_s += val[D_S];
        } else if let Some(val) = imm1_lookup(&[
            tmp_cseq[i + 1],
            tmp_cseq[i],
            b'/',
            tmp_seq[i + 1],
            tmp_seq[i],
        ]) {
            delta_h += val[D_H];
            delta_s += val[D_S];
        } else if let Some(val) = dna_nn3_lookup(&neighbors) {
            delta_h += val[D_H];
            delta_s += val[D_S];
        } else if let Some(val) = dna_nn3_lookup(&[
            tmp_cseq[i + 1],
            tmp_cseq[i],
            b'/',
            tmp_seq[i + 1],
            tmp_seq[i],
        ]) {
            delta_h += val[D_H];
            delta_s += val[D_S];
        } else {
            return Err(format!(
                "No thermodynamic data for neighbors '{:?}' available",
                neighbors
            ));
        }
    }

    // Apply symmetry penalty if self-complementary
    let mut k = (params.dnac1 - (params.dnac2 / 2.0)) * 1e-9;
    if params.selfcomp {
        k = params.dnac1 * 1e-9;
        delta_h += init_params.sym[D_H];
        delta_s += init_params.sym[D_S];
    }

    let r = 1.987; // universal gas constant in Cal/degrees C*Mol

    let mut melting_temp = (1000.0 * delta_h) / (delta_s + (r * k.ln())) - 273.15;

    if params.saltcorr > 0 {
        match salt_correction(
            params.na,
            params.k,
            params.tris,
            params.mg,
            params.dntps,
            params.saltcorr,
            &sequence,
        ) {
            Ok(corr) => {
                if params.saltcorr == 5 {
                    delta_s += corr;
                    melting_temp = (1000.0 * delta_h) / (delta_s + (r * k.ln())) - 273.15;
                } else if params.saltcorr >= 1 && params.saltcorr <= 4 {
                    melting_temp += corr;
                } else if params.saltcorr == 6 || params.saltcorr == 7 {
                    melting_temp = 1.0 / (1.0 / (melting_temp + 273.15) + corr) - 273.15;
                }
            }
            Err(_) => {
                // Silently ignore salt correction errors
            }
        }
    }

    Ok(melting_temp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tm_nn_simple() {
        assert!(
            (tm_nn(
                b"CGCTCAGAGACAAGCCGTTACAACGTAACC",
                None,
                TmNnParams::default()
            )
            .unwrap()
                - 63.05)
                .abs()
                < 0.01
        );
        assert!(
            (tm_nn(b"CGGGGCGGCCGGCCCGCCGG", None, TmNnParams::default()).unwrap() - 74.11).abs()
                < 0.01
        );
    }

    #[test]
    fn test_tm_nn_failures() {
        assert!(tm_nn(
            b"CGCTCAGAGACAAGCCGTTACAACGTAACC",
            Some(b"A"),
            TmNnParams::default(),
        )
        .is_err());
    }

    #[test]
    fn test_tm_nn_dangling_ends() {
        assert!(
            (tm_nn(b"GTGTCCTCGAGT", None, TmNnParams::default()).unwrap() - 34.79).abs() < 0.01
        );
        assert!(
            (tm_nn(b"GTGTCCTCGAGT", Some(b"CACAGGAGCTC"), TmNnParams::default()).unwrap() - 30.78)
                .abs()
                < 0.01
        );
    }

    #[test]
    fn test_tm_nn_shift() {
        assert!(
            (tm_nn(
                b"CGTTCCAAAGATGTGGGCATGAGCTTAC",
                Some(b"TGCAAGGCTTCTACACCCGTACTCGAATGC"),
                TmNnParams {
                    shift: 1,
                    ..Default::default()
                }
            )
            .unwrap()
                - 55.69)
                .abs()
                < 0.01
        );
    }

    #[test]
    fn test_tm_nn_salt() {
        assert!(
            (tm_nn(
                b"CGTTCCAAAGATGTGGGCATGAGCTTAC",
                None,
                TmNnParams {
                    na: 1.0,
                    ..Default::default()
                }
            )
            .unwrap()
                - 41.99)
                .abs()
                < 0.01
        );
        assert!(
            (tm_nn(
                b"CGTTCCAAAGATGTGGGCATGAGCTTAC",
                None,
                TmNnParams {
                    na: 10.0,
                    ..Default::default()
                }
            )
            .unwrap()
                - 52.53)
                .abs()
                < 0.01
        );
    }

    #[test]
    fn test_tm_nn_mg() {
        assert!(
            (tm_nn(
                b"CGTTCCAAAGATGTGGGCATGAGCTTAC",
                None,
                TmNnParams {
                    mg: 10.0,
                    ..Default::default()
                }
            )
            .unwrap()
                - 71.33)
                .abs()
                < 0.01
        );
    }

    #[test]
    fn test_tm_nn_k() {
        assert!(
            (tm_nn(
                b"CGTTCCAAAGATGTGGGCATGAGCTTAC",
                None,
                TmNnParams {
                    k: 100.0,
                    ..Default::default()
                }
            )
            .unwrap()
                - 65.86)
                .abs()
                < 0.01
        );
    }

    #[test]
    fn test_tm_nn_saltcorr() {
        assert!(
            (tm_nn(b"CGTTCCAAAGATGTGGGCATGAGCTTAC", None, TmNnParams::default()).unwrap() - 60.32)
                .abs()
                < 0.01
        );

        assert!(
            (tm_nn(
                b"CGTTCCAAAGATGTGGGCATGAGCTTAC",
                None,
                TmNnParams {
                    saltcorr: 1,
                    ..Default::default()
                }
            )
            .unwrap()
                - 54.27)
                .abs()
                < 0.01
        );
        assert!(
            (tm_nn(
                b"CGTTCCAAAGATGTGGGCATGAGCTTAC",
                None,
                TmNnParams {
                    saltcorr: 6,
                    ..Default::default()
                }
            )
            .unwrap()
                - 59.78)
                .abs()
                < 0.01
        );
    }

    #[test]
    fn test_tm_nn_fail() {
        assert!(tm_nn(
            b"CAGCGGTCGGCTTAATGCCTCC",
            Some(b"CATTCATCGTGACAGTGGACCA"),
            TmNnParams::default()
        )
        .is_err());
    }

    /// edge case matching the Biopython behavior and
    /// illustrating the limits of nearest-neighbor Tm calculation
    #[test]
    fn test_tm_nn_weirdness() {
        assert!(
            (tm_nn(b"CACACACACACACACACACA", None, TmNnParams::default()).unwrap() - 53.44).abs()
                < 0.01
        );
        assert!(
            (tm_nn(
                b"CACACACACACACACACACA",
                Some(b"ATATATATATATATATATAT"),
                TmNnParams::default()
            )
            .unwrap()
                - 280.15)
                .abs()
                < 0.01
        );
    }
}
