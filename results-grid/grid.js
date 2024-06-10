agGrid.LicenseManager.setLicenseKey("YOUR_LICENSE_KEY_HERE");

document.addEventListener('DOMContentLoaded', function () {
    const gridOptions = {
        columnDefs: [
            {
                headerName: "", children: [
                    {field: 'TEST_ID', rowDrag: true, filter: 'agNumberColumnFilter', checkboxSelection: true,},
                ]
            },
            {
                headerName: "TEST INSTANCES", children: [
                    {headerName: "FILE NAME", field: "FILE_NAME", rowGroup: true,},
                    {headerName: "VEHICLE COUNT", field: "VEHICLE_COUNT", filter: 'agNumberColumnFilter',},
                    {headerName: "TROLLEY COUNT", field: "TROLLEY_COUNT", filter: 'agNumberColumnFilter',},
                    {
                        headerName: "TROLLEY IMPACT TIME",
                        field: "TROLLEY_IMPACT_TIME",
                        sortable: true,
                        filter: 'agNumberColumnFilter',

                    },
                    {
                        headerName: "EARLINESS/TARDINESS PENALTY",
                        field: "EARLINESSTARDINESS_PENALTY",
                        sortable: true,
                        filter: 'agNumberColumnFilter',
                    },
                ]
            },
            {
                headerName: "MATHEMATICAL FORMULATION", children: [
                    {
                        headerName: "Optimal OR Feasible",
                        field: "Math_Form_Is_Optimal_OR_Feasible",
                        sortable: true,
                        filter: true,

                    },
                    {headerName: "Math CPU TIME", field: "Math_CPU_TIME", filter: 'agNumberColumnFilter',},
                    {headerName: "Math RESULT", field: "Math_RESULT", filter: 'agNumberColumnFilter',},
                    {headerName: "Math Distance Cost", field: "Math_Model_Distance_Cost", filter: 'agNumberColumnFilter',},
                    {headerName: "Math EA Cost", field: "Math_Model_EA_Cost", filter: 'agNumberColumnFilter',},
                    {headerName: "Math TA Cost", field: "Math_Model_TA_Cost", filter: 'agNumberColumnFilter',},
                ]
            },
            {
                headerName: "ALNS", children: [
                    {headerName: "ALNS CPU TIME", field: "ALNS_CPU_TIME", filter: 'agNumberColumnFilter',},
                    {headerName: "ALNS RESULT", field: "ALNS_RESULT", filter: 'agNumberColumnFilter',},
                    {headerName: "ALNS Distance Cost", field: "ALNS_Distance_Cost", filter: 'agNumberColumnFilter',},
                    {headerName: "ALNS EA Cost", field: "ALNS_EA_Cost", filter: 'agNumberColumnFilter',},
                    {headerName: "ALNS TA Cost", field: "ALNS_TA_Cost", filter: 'agNumberColumnFilter',},
                ]
            },
            {
                headerName: "Rule Of Thumb", children: [
                    {headerName: "ROTH CPU TIME", field: "ROTH_CPU_TIME", filter: 'agNumberColumnFilter',},
                    {headerName: "ROTH RESULT", field: "ROTH_RESULT", filter: 'agNumberColumnFilter',},
                    {headerName: "ROTH Distance Cost", field: "ROTH_Distance_Cost", filter: 'agNumberColumnFilter',},
                    {headerName: "ROTH EA Cost", field: "ROTH_EA_Cost", filter: 'agNumberColumnFilter',},
                    {headerName: "ROTH TA Cost", field: "ROTH_TA_Cost", filter: 'agNumberColumnFilter',},
                ]
            },
            {
                headerName: "", children: [
                    {headerName: "GAP", field: "GAP", filter: 'agNumberColumnFilter',}
                ]
            },


        ], defaultColDef: {
            sortable: true,
            floatingFilter: true,
            pivot: true,
            enableRowGroup: true,
            enableValue: true,
            enablePivot: true,
            flex: 1,
        },
        groupDisplayType: "multipleColumns",
        rowDragManaged: true,
        rowSelection: "multiple",
        autoGroupColumnDef: {
            filter: "agGroupColumnFilter",
        },
        enableAdvancedFilter: true,
        rowGroupPanelShow: "always",
        enableRangeSelection: true,
        sideBar: ["columns", "filters"],
        statusBar: {
            statusPanels: [
                {statusPanel: "agTotalAndFilteredRowCountComponent"},
                {statusPanel: "agTotalRowCountComponent"},
                {statusPanel: "agFilteredRowCountComponent"},
                {statusPanel: "agSelectedRowCountComponent"},
                {statusPanel: "agAggregationComponent"},
            ],
        },
         enableCharts: true,
        rowData: rowData
    };

    const eGridDiv = document.querySelector('#myGrid');
    new agGrid.Grid(eGridDiv, gridOptions);
});