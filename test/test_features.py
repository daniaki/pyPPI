from pyPPI.data_mining.uniprot import get_active_instance
from pyPPI.data_mining.features import AnnotationExtractor

from unittest import TestCase


class TestAnnotationExtractor(TestCase):

    def setUp(self):
        self.uniprot = get_active_instance(
            sprot_cache=None,
            trembl_cache=None,
            allow_download=True
        )
        self.ppis_a = [
            ('P20138', 'Q9UBI1'),
            ('O75528', 'P10276')
        ]
        self.ppis_b = [
            ('Q9UQC1', 'Q9Y6K9'),
            ('P16234', 'Q9GZP0')
        ]
        self.ppis_non_human = [
            ('P20138', 'Q9UBI1'),
            ('Q8R2R3', 'Q91V24')  # This is from Mus musculus (Mouse)
        ]

    def test_can_fit_ppi_list(self):
        ae = AnnotationExtractor(
            induce=False,
            n_jobs=1,
            selection=[
                self.uniprot.data_types().GO_BP.value,
                self.uniprot.data_types().GO_CC.value,
                self.uniprot.data_types().GO_MF.value,
                self.uniprot.data_types().INTERPRO.value,
                self.uniprot.data_types().PFAM.value,
            ]
        )
        ae.fit(self.ppis_a)
        self.assertEqual(len(ae.accession_vocabulary.index), 4)
        self.assertEqual(len(ae.ppi_vocabulary.index), 2)

    def test_correct_features_returned(self):
        ae = AnnotationExtractor(
            induce=False,
            n_jobs=1,
            selection=[
                self.uniprot.data_types().GO_BP.value,
                self.uniprot.data_types().GO_CC.value,
            ]
        )
        X = ae.fit_transform(self.ppis_a)
        df = ae.accession_vocabulary
        column_key = self.uniprot.accession_column()

        ppi_1_X = ','.join(sorted(
            df[df[column_key] == self.ppis_a[0][0]][
                self.uniprot.data_types().GO_BP.value
            ].values[0] +
            df[df[column_key] == self.ppis_a[0][0]][
                self.uniprot.data_types().GO_CC.value
            ].values[0] +

            df[df[column_key] == self.ppis_a[0][1]][
                self.uniprot.data_types().GO_BP.value
            ].values[0] +
            df[df[column_key] == self.ppis_a[0][1]][
                self.uniprot.data_types().GO_CC.value
            ].values[0]
        ))
        ppi_2_X = ','.join(sorted(
            df[df[column_key] == self.ppis_a[1][0]][
                self.uniprot.data_types().GO_BP.value
            ].values[0] +
            df[df[column_key] == self.ppis_a[1][0]][
                self.uniprot.data_types().GO_CC.value
            ].values[0] +

            df[df[column_key] == self.ppis_a[1][1]][
                self.uniprot.data_types().GO_BP.value
            ].values[0] +
            df[df[column_key] == self.ppis_a[1][1]][
                self.uniprot.data_types().GO_CC.value
            ].values[0]
        ))
        self.assertEqual(
            ','.join(sorted(X[0].split(','))), ppi_1_X.replace(':', ''))
        self.assertEqual(
            ','.join(sorted(X[1].split(','))), ppi_2_X.replace(':', ''))

    def test_can_refit_to_new_ppi_list(self):
        ae = AnnotationExtractor(
            induce=False,
            n_jobs=1,
            selection=[
                self.uniprot.data_types().GO_BP.value,
            ]
        ).fit(self.ppis_a)
        df_1a = ae.accession_vocabulary
        df_2a = ae.ppi_vocabulary

        ae.fit(self.ppis_b)
        df_1b = ae.accession_vocabulary
        df_2b = ae.ppi_vocabulary

        self.assertFalse(df_1a.equals(df_2a))
        self.assertFalse(df_1b.equals(df_2b))

    def test_raise_value_error_if_transform_on_unseen_ppis(self):
        ae = AnnotationExtractor(
            induce=False,
            n_jobs=1,
            selection=[
                self.uniprot.data_types().GO_BP.value,
            ]
        ).fit(self.ppis_a)
        with self.assertRaises(ValueError):
            ae.transform(self.ppis_b)

    def test_can_identify_invalid_ppis_missing_data(self):
        ae = AnnotationExtractor(
            induce=False,
            n_jobs=1,
            selection=[
                self.uniprot.data_types().GO_BP.value,
            ]
        ).fit(self.ppis_non_human)

        invalid = ae.invalid_ppis(self.ppis_non_human)
        print(invalid)
        self.assertEqual(invalid, [self.ppis_non_human[1]])

    def test_transforms_return_same_order(self):
        ae = AnnotationExtractor(
            induce=False,
            n_jobs=1,
            selection=[
                self.uniprot.data_types().GO_BP.value,
            ]
        )
        X = ae.fit_transform(self.ppis_non_human)
        self.assertEqual(X[1], '')

    def test_valueerror_non_ppi_tuple_input(self):
        with self.assertRaises(ValueError):
            ae = AnnotationExtractor(
                induce=False,
                n_jobs=1,
                selection=[
                    self.uniprot.data_types().GO_BP.value,
                ]
            ).fit([self.ppis_a[0][0]])
