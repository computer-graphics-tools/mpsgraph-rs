use std::fmt;
use std::str::FromStr;

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct ComputeDevice: u64 {
        const GPU = 1 << 0;
        const ANE = 1 << 1;
        const CPU = 1 << 2;
    }
}

impl fmt::Display for ComputeDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return write!(f, "unknown");
        }
        let mut parts = Vec::new();
        if self.contains(ComputeDevice::GPU) {
            parts.push("gpu");
        }
        if self.contains(ComputeDevice::ANE) {
            parts.push("ane");
        }
        if self.contains(ComputeDevice::CPU) {
            parts.push("cpu");
        }
        write!(f, "{}", parts.join("+"))
    }
}

impl FromStr for ComputeDevice {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s == "unknown" {
            // Or handle as an error, depending on desired behavior for "unknown"
            return Ok(ComputeDevice::empty());
        }
        let mut device = ComputeDevice::empty();
        for part in s.split('+') {
            match part {
                "gpu" => device |= ComputeDevice::GPU,
                "ane" => device |= ComputeDevice::ANE,
                "cpu" => device |= ComputeDevice::CPU,
                _ => return Err(format!("Invalid device string: {}", part)),
            }
        }
        if device.is_empty() && !s.is_empty() && s != "unknown" {
            Err(format!(
                "Invalid or empty MPSGraphComputeDevice string: {}",
                s
            ))
        } else {
            Ok(device)
        }
    }
}
