/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef __DOCA_BACKEND_H
#define __DOCA_BACKEND_H

#include <vector>
#include <cstring>
#include <iostream>
#include <thread>
#include <mutex>
#include <atomic>
#include <sys/types.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <doca_ctx.h>
#include <doca_dev.h>
#include <doca_error.h>
#include <doca_log.h>
#include <doca_gpunetio.h>
#include <doca_pe.h>
#include <doca_rdma.h>
#include <doca_rdma_bridge.h>
#include <doca_mmap.h>
#include <doca_buf_array.h>

#include "nixl.h"
#include "backend/backend_engine.h"
#include "public/backend_plugin_doca_common.h"
#include "common/str_tools.h"

// Local includes
#include "common/nixl_time.h"
#include "common/list_elem.h"

#define DOCA_DEVINFO_IBDEV_NAME_SIZE 64
#define RDMA_RECV_QUEUE_SIZE 2048
#define RDMA_SEND_QUEUE_SIZE 2048
#define DOCA_XFER_REQ_SIZE 512
#define DOCA_XFER_REQ_MAX 16
#define DOCA_ENG_MAX_CONN 10
#define DOCA_RDMA_CM_LOCAL_PORT 1235
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define DOCA_RDMA_SERVER_ADDR_LEN (MAX(MAX(DOCA_DEVINFO_IPV4_ADDR_SIZE, DOCA_DEVINFO_IPV6_ADDR_SIZE), DOCA_GID_BYTE_LENGTH))
#define DOCA_RDMA_SERVER_CONN_DELAY 500 //500us

struct nixlDocaMem {
    void *addr;
    uint32_t len;
    struct doca_mmap *mmap;
    void *export_mmap;
    size_t export_len;
    struct doca_buf_arr *barr;
    struct doca_gpu_buf_arr *barr_gpu;
    uint32_t devId;
};

class nixlDocaConnection : public nixlBackendConnMD {
    private:
        std::string remoteAgent;
        // rdma qp
        // nixlDocaEp ep;
        volatile bool connected;

    public:
        // Extra information required for UCX connections

    friend class nixlDocaEngine;
};

// A private metadata has to implement get, and has all the metadata
class nixlDocaPrivateMetadata : public nixlBackendMD {
    private:
        nixlDocaMem mem;
        nixl_blob_t remoteMmapStr;

    public:
        nixlDocaPrivateMetadata() : nixlBackendMD(true) {
        }

        ~nixlDocaPrivateMetadata(){
        }

        std::string get() const {
            return remoteMmapStr;
        }

    friend class nixlDocaEngine;
};

// A public metadata has to implement put, and only has the remote metadata
class nixlDocaPublicMetadata : public nixlBackendMD {

    public:
        nixlDocaMem mem;
        nixlDocaConnection conn;

        nixlDocaPublicMetadata() : nixlBackendMD(false) {}

        ~nixlDocaPublicMetadata(){
        }
};

class nixlDocaEngine : public nixlBackendEngine {
    private:
        std::vector<std::pair<uint32_t, struct doca_gpu *>> gdevs; /* List of DOCA GPUNetIO device handlers */
	    struct doca_dev *ddev;	  /* DOCA device handler associated to queues */
        struct doca_log_backend *sdk_log;
        struct doca_rdma *rdma;		    /* DOCA RDMA instance */
        struct doca_gpu_dev_rdma *rdma_gpu; /* DOCA RDMA instance GPU handler */
        struct doca_ctx *rdma_ctx;	    /* DOCA context to be used with DOCA RDMA */
        struct doca_pe *pe;	    /* DOCA progress engine for RDMA CM mode */
        uint32_t local_port;
        int noSyncIters;
        //Change it
        std::atomic<uint32_t> connection_num;
        uint32_t last_connection_num;
        union doca_data ctx_user_data;
        struct doca_rdma_connection *connection[DOCA_ENG_MAX_CONN];
        volatile bool connection_error;
        // const void *connection_details;	    /* Remote peer connection details */
        // const uint8_t connection_details[DOCA_RDMA_SERVER_ADDR_LEN];	    /* Remote peer connection details */
        uint8_t ipv4_addr[4];
        size_t conn_det_len;		    /* Remote peer connection details data length */
        volatile uint8_t connection_established[DOCA_ENG_MAX_CONN];
        struct doca_rdma_addr *cm_addr;		/* RDMA_CM server IPv4/IPv6/GID address */
        enum doca_rdma_addr_type cm_addr_type;		/* RDMA_CM server address type, IPv4, IPv6 or GID. Only useful for client */

        std::thread pthr;
        volatile bool pthrStop, pthrActive;

        struct docaXferReqGpu *xferReqRingGpu;
        struct docaXferReqGpu *xferReqRingCpu;
        std::atomic<uint32_t> xferRingPos;
        uint32_t firstXferRingPos;
        // Map of agent name to saved nixlDocaConnection info
        std::unordered_map<std::string, nixlDocaConnection,
                           std::hash<std::string>, strEqual> remoteConnMap;

        class nixlDocaBckndReq : public nixlLinkElem<nixlDocaBckndReq>, public nixlBackendReqH {
            private:
            public:
                cudaStream_t stream;
                uint32_t devId;
                uint32_t start_pos;
                uint32_t end_pos;
                uintptr_t backendHandleGpu;

                nixlDocaBckndReq() : nixlLinkElem(), nixlBackendReqH() {
                }

                ~nixlDocaBckndReq() {
                }
        };

        // Request management
        static void _requestInit(void *request);
        static void _requestFini(void *request);
        void requestReset(nixlDocaBckndReq *req) {
            _requestInit((void *)req);
        }

        // Memory management helpers
        nixl_status_t internalMDHelper (const nixl_blob_t &blob,
                                        const std::string &agent,
                                        nixlBackendMD* &output);

        // Threading infrastructure
        //   TODO: move the thread management one outside of NIXL common infra
        void progressFunc();
        void progressThreadStart();
        void progressThreadStop();
        void progressThreadRestart();

    public:
        nixlDocaEngine(const nixlBackendInitParams* init_params);
        ~nixlDocaEngine();

        bool supportsRemote () const { return true; }
        bool supportsLocal () const { return false; }
        bool supportsNotif () const { return false; }
        bool supportsProgTh () const { return false; }
        bool supportsGpuInitiated () const { return true; }

        nixl_mem_list_t getSupportedMems () const;

        /* Object management */
        nixl_status_t getPublicData (const nixlBackendMD* meta,
                                     std::string &str) const;
        nixl_status_t getConnInfo(std::string &str) const;
        nixl_status_t loadRemoteConnInfo (const std::string &remote_agent,
                                          const std::string &remote_conn_info);

        nixl_status_t connect(const std::string &remote_agent);
        nixl_status_t disconnect(const std::string &remote_agent);

        nixl_status_t registerMem (const nixlBlobDesc &mem,
                                   const nixl_mem_t &nixl_mem,
                                   nixlBackendMD* &out);
        nixl_status_t deregisterMem (nixlBackendMD* meta);

        nixl_status_t loadLocalMD (nixlBackendMD* input,
                                   nixlBackendMD* &output);

        nixl_status_t loadRemoteMD (const nixlBlobDesc &input,
                                    const nixl_mem_t &nixl_mem,
                                    const std::string &remote_agent,
                                    nixlBackendMD* &output);
        nixl_status_t unloadMD (nixlBackendMD* input);

        // Data transfer
        nixl_status_t prepXfer (const nixl_xfer_op_t &operation,
                                const nixl_meta_dlist_t &local,
                                const nixl_meta_dlist_t &remote,
                                const std::string &remote_agent,
                                nixlBackendReqH* &handle,
                                const nixl_opt_b_args_t* opt_args=nullptr);

        nixl_status_t postXfer (const nixl_xfer_op_t &operation,
                                const nixl_meta_dlist_t &local,
                                const nixl_meta_dlist_t &remote,
                                const std::string &remote_agent,
                                nixlBackendReqH* &handle,
                                const nixl_opt_b_args_t* opt_args=nullptr);

        nixl_status_t checkXfer (nixlBackendReqH* handle);
        nixl_status_t releaseReqH(nixlBackendReqH* handle);

        nixl_status_t getGpuXferH(const nixlBackendReqH* handle, nixlXferReqHGpu* &gpu_hndl);

        int progress();

        nixl_status_t getNotifs(notif_list_t &notif_list);
        nixl_status_t genNotif(const std::string &remote_agent, const std::string &msg);

        void addConnection(struct doca_rdma_connection *connection);
        uint32_t getConnectionLast();
        void removeConnection(uint32_t connection_idx);
        uint32_t getGpuCudaId();
};

doca_error_t doca_util_map_and_export(struct doca_dev *dev, uint32_t permissions, void *addr, uint32_t size, nixlDocaMem *mem);

#if __cplusplus
extern "C" {
#endif

// prepXferGpu postXferGpuGet();
doca_error_t doca_kernel_write(cudaStream_t stream, struct doca_gpu_dev_rdma *rdma_gpu, struct docaXferReqGpu *xferReqRing, uint32_t pos);
doca_error_t doca_kernel_read(cudaStream_t stream, struct doca_gpu_dev_rdma *rdma_gpu, struct docaXferReqGpu *xferReqRing, uint32_t pos);

#if __cplusplus
}
#endif

#endif
