import tempfile

from pycdlib import pycdlib


def create_cloud_init_iso(iso_path, user_data_fn, meta_data_fn):
    with tempfile.NamedTemporaryFile(
        mode="w"
    ) as user_data_file, tempfile.NamedTemporaryFile(
        mode="w"
    ) as meta_data_file:
        user_data_file.write(user_data_fn())
        user_data_file.flush()

        meta_data_file.write(meta_data_fn())
        meta_data_file.flush()

        _create_cloud_init_iso_from_files(
            iso_path, user_data_file.name, meta_data_file.name
        )


def _create_cloud_init_iso_from_files(iso_path, user_data_path, meta_data_path):
    iso = pycdlib.PyCdlib()
    iso.new(vol_ident="cidata", joliet=True, rock_ridge="1.09")

    iso.add_file(
        user_data_path,
        "/USERDATA.;1",
        rr_name="user-data",
        joliet_path="/user-data",
    )
    iso.add_file(
        meta_data_path,
        "/METADATA.;1",
        rr_name="meta-data",
        joliet_path="/meta-data",
    )

    iso.write(iso_path)
    iso.close()
