#!/usr/bin/env python3
"""Parse Bynav RAWIMUSX binary packets into CSV."""

from __future__ import annotations

import argparse
import csv
import gzip
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, TextIO


SYNC = b"\xaa\x44"
SHORT_BINARY_TYPE = 0x13
RAWIMUSX_MESSAGE_ID = 1462
RAWIMUSX_PAYLOAD_LENGTH = 40
RAWIMUSX_PACKET_LENGTH = 56  # 12-byte short header + 40-byte payload + 4-byte CRC
RAWIMUSX_SIGNATURE = b"\xaa\x44\x13\x28\xb6\x05"
GPS_EPOCH_UNIX = 315964800
GPS_LEAP_SECONDS = 18


@dataclass(frozen=True)
class ImuScale:
    accel_dv_mps_per_lsb: float
    gyro_dangle_deg_per_lsb: float
    rate_hz: float


IMU_SCALES: dict[int, ImuScale] = {
    3: ImuScale(
        accel_dv_mps_per_lsb=4.65661287307739e-08,
        gyro_dangle_deg_per_lsb=3.35276126861572e-07,
        rate_hz=100.0,
    ),
    5: ImuScale(
        accel_dv_mps_per_lsb=2.99275207519531e-08,
        gyro_dangle_deg_per_lsb=2.44140625e-07,
        rate_hz=125.0,
    ),
    6: ImuScale(
        accel_dv_mps_per_lsb=5.98550415039063e-08,
        gyro_dangle_deg_per_lsb=2.31193542480469e-07,
        rate_hz=125.0,
    ),
}


@dataclass(frozen=True)
class RawImuSample:
    week: int
    seconds: float
    imu_info: int
    imu_type: int
    imu_status: int
    z_accel_lsb: int
    neg_y_accel_lsb: int
    x_accel_lsb: int
    z_gyro_lsb: int
    neg_y_gyro_lsb: int
    x_gyro_lsb: int


def iter_rawimusx_packets(data: bytes) -> Iterator[RawImuSample]:
    offset = 0
    data_length = len(data)

    while True:
        start = data.find(RAWIMUSX_SIGNATURE, offset)
        if start < 0 or start + RAWIMUSX_PACKET_LENGTH > data_length:
            return

        packet = data[start : start + RAWIMUSX_PACKET_LENGTH]
        body = packet[12:52]

        yield RawImuSample(
            imu_info=body[0],
            imu_type=body[1],
            week=struct.unpack_from("<H", body, 2)[0],
            seconds=struct.unpack_from("<d", body, 4)[0],
            imu_status=struct.unpack_from("<I", body, 12)[0],
            z_accel_lsb=struct.unpack_from("<i", body, 16)[0],
            neg_y_accel_lsb=struct.unpack_from("<i", body, 20)[0],
            x_accel_lsb=struct.unpack_from("<i", body, 24)[0],
            z_gyro_lsb=struct.unpack_from("<i", body, 28)[0],
            neg_y_gyro_lsb=struct.unpack_from("<i", body, 32)[0],
            x_gyro_lsb=struct.unpack_from("<i", body, 36)[0],
        )

        offset = start + RAWIMUSX_PACKET_LENGTH


def format_float(value: float) -> str:
    return f"{value:.12f}"


def open_csv_writer(output_path: Path) -> tuple[TextIO, csv.writer]:
    if output_path.suffix == ".gz":
        handle = gzip.open(output_path, "wt", newline="")
    else:
        handle = output_path.open("w", newline="")
    return handle, csv.writer(handle)


def build_row(sample: RawImuSample, scale: ImuScale) -> list[str | int]:
    dt = 1.0 / scale.rate_hz

    z_dv = sample.z_accel_lsb * scale.accel_dv_mps_per_lsb
    neg_y_dv = sample.neg_y_accel_lsb * scale.accel_dv_mps_per_lsb
    x_dv = sample.x_accel_lsb * scale.accel_dv_mps_per_lsb
    y_dv = -neg_y_dv

    z_accel = z_dv / dt
    neg_y_accel = neg_y_dv / dt
    x_accel = x_dv / dt
    y_accel = -neg_y_accel

    z_dangle = sample.z_gyro_lsb * scale.gyro_dangle_deg_per_lsb
    neg_y_dangle = sample.neg_y_gyro_lsb * scale.gyro_dangle_deg_per_lsb
    x_dangle = sample.x_gyro_lsb * scale.gyro_dangle_deg_per_lsb
    y_dangle = -neg_y_dangle

    z_rate = z_dangle / dt
    neg_y_rate = neg_y_dangle / dt
    x_rate = x_dangle / dt
    y_rate = -neg_y_rate

    unix_time_gps = GPS_EPOCH_UNIX + sample.week * 604800 + sample.seconds
    unix_time_utc_est = unix_time_gps - GPS_LEAP_SECONDS

    return [
        sample.week,
        f"{sample.seconds:.8f}",
        f"{unix_time_gps:.8f}",
        f"{unix_time_utc_est:.8f}",
        sample.imu_info,
        sample.imu_type,
        sample.imu_status,
        sample.z_accel_lsb,
        sample.neg_y_accel_lsb,
        sample.x_accel_lsb,
        sample.z_gyro_lsb,
        sample.neg_y_gyro_lsb,
        sample.x_gyro_lsb,
        format_float(z_dv),
        format_float(neg_y_dv),
        format_float(x_dv),
        format_float(y_dv),
        format_float(z_accel),
        format_float(neg_y_accel),
        format_float(x_accel),
        format_float(y_accel),
        format_float(z_dangle),
        format_float(neg_y_dangle),
        format_float(x_dangle),
        format_float(y_dangle),
        format_float(z_rate),
        format_float(neg_y_rate),
        format_float(x_rate),
        format_float(y_rate),
    ]


def parse_log(input_path: Path, output_path: Path, limit: int | None) -> tuple[int, RawImuSample | None, RawImuSample | None]:
    data = input_path.read_bytes()
    count = 0
    first_sample: RawImuSample | None = None
    last_sample: RawImuSample | None = None

    header = [
        "week",
        "seconds",
        "unix_time_gps",
        "unix_time_utc_est",
        "imu_info",
        "imu_type",
        "imu_status",
        "z_accel_lsb",
        "neg_y_accel_lsb",
        "x_accel_lsb",
        "z_gyro_lsb",
        "neg_y_gyro_lsb",
        "x_gyro_lsb",
        "z_dv_mps",
        "neg_y_dv_mps",
        "x_dv_mps",
        "y_dv_mps",
        "z_accel_mps2",
        "neg_y_accel_mps2",
        "x_accel_mps2",
        "y_accel_mps2",
        "z_dangle_deg",
        "neg_y_dangle_deg",
        "x_dangle_deg",
        "y_dangle_deg",
        "z_gyro_dps",
        "neg_y_gyro_dps",
        "x_gyro_dps",
        "y_gyro_dps",
    ]

    handle, writer = open_csv_writer(output_path)
    try:
        writer.writerow(header)
        for sample in iter_rawimusx_packets(data):
            scale = IMU_SCALES.get(sample.imu_type)
            if scale is None:
                supported = ", ".join(str(imu_type) for imu_type in sorted(IMU_SCALES))
                raise ValueError(
                    f"Unsupported IMU type {sample.imu_type}. "
                    f"Supported types: {supported}."
                )

            writer.writerow(build_row(sample, scale))
            count += 1

            if first_sample is None:
                first_sample = sample
            last_sample = sample

            if limit is not None and count >= limit:
                break
    finally:
        handle.close()

    return count, first_sample, last_sample


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse Bynav RAWIMUSX packets from a KD_*BY.LOG file."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="KD_202404121052208133BY.LOG",
        help="Input binary log file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="rawimusx.csv.gz",
        help="Output CSV file. Use .gz suffix for gzip compression.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only export the first N RAWIMUSX packets.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    count, first_sample, last_sample = parse_log(input_path, output_path, args.limit)

    print(f"output={output_path}")
    print(f"rows={count}")
    if first_sample is not None:
        print(f"first={first_sample.week},{first_sample.seconds:.8f}")
    if last_sample is not None:
        print(f"last={last_sample.week},{last_sample.seconds:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
